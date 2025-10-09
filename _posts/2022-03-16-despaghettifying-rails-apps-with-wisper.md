---
title: "De-spaghettifying Rails Apps with Wisper"
layout: post
date: 2022-3-16 17:21
image: '/assets/images/'
description:
tag: rails wisper
blog: true
jemoji:
author:
---

Let’s say we have a Rails application that users can sign up to, and we want to add a feature to send new users a welcome email on registration. Where should we put that logic?

## Option 1: The controller

We could put it inside our call to create the user in the user registration controller.

Something like this...

```ruby
class Registrations
	def create
		user = User.new(user_params)

		if user.save?
			# Send the welcome email if a user is saved successfully
			UserMailer.with(user: resource).welcome_email.deliver_later
			redirect_to root_path, notice: "Signed up!"
		else
			redirect_to root_path, notice: "Could not create account"
		end
	end
end
```

I’d argue that putting this logic in the controller is fine for small things, but it’s not the best solution.

If our app starts to grow and we need to do more things like store a “sign up event” to our database, or send a notification to slack to say we’ve acquired a new user, then our controller starts to bloat pretty quickly with a lot of non-registration related logic.

## Option #2: Callbacks

We could use an `after_create` callback in our User model, but I like this even less. Creating users in the console for test purposes would fire off an unwanted welcome email, and we’re coupling mailer code tightly to our model.

Shoehorning these things into the controller feels messy, and they don’t feel at home in our models either.

So what’s the solution?

## Option #3: Pub/Sub style events with Wisper

Wisper is a minimalist ruby library that allows us to broadcast events, and listen for them somewhere else in our codebase.

Wisper gives us a simple pattern for dealing with this problem by decoupling code and just passing messages around instead.

Whenever something happens that we want to care about, like the creation of a new user, we’ll send out a `:user_created` event and have a listener somewhere else that picks up these events, and sends the mailer.

## Installation

Let’s start by installing wisper in our gemfile...

```ruby
gem 'wisper'
```

## Events

Wisper usually passes around raw data, but I prefer to create classes for specific events. Let’s create an event for our user creation. I tend to put these in `app/lib/events` or `app/models/events`.

```ruby
class Events::UserCreated
	attr_reader :user

	def initializer(user:)
		@user = user
	end
end
```

## Broadcasting an Event

To broadcast an event, we need to include `Wisper::Publisher` in the code we want to broadcast from. We’re broadcasting from the model, but we can use the same include to broadcast from a controller or anywhere else.

```ruby
class User < ApplicationRecord
	include Wisper::Publisher

	has_one :address

	after_create :broadcast_user_created_event

	private

	def broadcast_user_created_event
		broadcast(:user_created, Events::UserCreated.new(user: self))
	end
end
```

Wisper’s `broadcast` method takes two arguments, the first is the event name as a symbol, the second is the payload. This could be a hash, string or any bit of data, but this is where using classes for events really pays off.

## Listening and Responding to Events

Once we’ve got our events, we’ll need to create a Listener to respond to them. I like to put listeners in `app/lib/listeners` (more about naming conventions later)

```ruby
class Listeners::UserListener
  def on_user_created(event)
		UserMailer.with(user: event.user).welcome_email.deliver_late
  end
end
```

Our listener should define methods with the same name as the event name. In this case, our event is called `:user_created`, so we should define a method called `on_user_created` that accept a single argument; the event payload we passed in to our `broadcast` method.

“Wait where does `on_` come from?”

Good question. It’s a stylistic choice, you don’t have to have the prefix, but I prefer it. It happens when you subscribe your listener, which we’ll cover next...

## Subscribe your Listener to Events

Our Listener doesn’t automatically pick up events unfortunately, we need to subscribe our listener. We’ll do this in an initializer file...

```ruby
Rails.application.config.to_prepare do
  # Wisper subscribers need to be refreshed here when we are in
  # dev/test. This is due to code-reloading, which could re-subscribe
  # existing handlers, leading to duplicates and errors
  Wisper.clear if Rails.env.development? || Rails.env.test?

  # Subscribe your listeners here, use prefix: :on to get event names like on_fund_created in the listener
  Wisper.subscribe(Listeners::FundListener.new, prefix: :on)
end
```

Here we’re setting the `prefix: :on` option, which changes the incoming method in our listener from `user_created` to `on_user_created`. This is a matter of preference, but I think it reads nicer with prefixes enabled.

## Done!

Your new events bus is now wired up and ready to go! Creating a user should now emit an event that gets picked up by your listener and fires off a welcome email.

It’s a little setup, but the reward for decoupling this code pays off, especially when you start dealing with a few different events.

## Some useful conventions

Here’s some conventions I’ve found useful to help to keep events stuff organised. I tend to document these in the project README too for other developers to follow.

I like to use classes with keyword arguments for events to give them a defined and documented structure. You can also use Structs or Dry Struct to further enforce events to have required attributes and formats.

I create two folders to house all my wisper stuff; `app/lib/events/` to house my event classes, and `app/lib/listeners/` to house the corresponding listeners. (Although you could move events to `models/events/` if you needed to persist some to the database)

I name events `Events::ThingVerb`, where `Thing` is usually the model name, and `Verb` is the past tense action that’s happening to it (created, updated, committed, etc), but feel free to adopt a convention that makes sense for your app, and then document it in your README.

This is what my file structure looks like:

```ruby
/app
	/lib
		/events
			user_created.rb
			user_updated.rb
		/listeners
			user_listener.rb
		/publishers
			events_publisher.rb
```

When subscribing a listener, I prefer using the `prefix: :on` option, so that events arrive at my listener with the naming convention `on_user_created`. I think it reads a bit better than the raw event name.

When using callbacks like `after_save`, I like to hand these off to a method with the convention `broadcast_event_name_event`, for example: `broadcast_user_created_event`. This helps create a consistent naming between my events, listeners, and anything calling them.

## Publishers for easier calling

“But passing in the event name as a symbol and the event object to `broadcast` feels like duplicating effort, can’t we just pass the event on its own?”

Yes! I’ve been using a pattern that allows us to just broadcast the event object itself.

```ruby
module Publishers
  module EventPublisher
    include Wisper::Publisher
    extend self

    alias_method :wisper_broadcast, :broadcast

    def broadcast(event)
      wisper_broadcast(symbolize_event(event), event)
    end

    def symbolize_event(event)
      event.class.name.demodulize.underscore.to_sym
    end
  end
end
```

Instead of adding `include Wisper::Publisher` in the file you want to broadcast from, you can now use `include Publishers::EventPublisher` instead, and broadcast your events like this:

```ruby
broadcast(Events::UserCreated.new(user: self))
```

Our `Publishers::EventPublisher` will take the class and pull the event from the demodulized class name, converting the `UserCreated` bit to `:user_created`.

Now we’re protected from accidentally misspelling an event.

## Bubbling events up from Child models

Let’s say our User model `has_many` Addresses. Can we get our `:user_updated` event to emit if the address is updated?

Getting a callback to fire on the user whenever the address is updated is actually quite simple, but comes with a gotcha.

ActiveRecord has a handy option that we can pass to `belongs_to` called `touch: true`.

Enabling touch means that whenever our child model changes, we’ll bump the `updated_at` timestamp on our parent model.

This is useful if you have a parent model where updates to the children should also be reflected in the parent, like a user profile where address is a separate model.

But the gotcha here is that `touch` does NOT perform validations, and will only trigger `after_commit`, `after_touch`, and `after_rollback` callbacks.

The best course of action is to use `belongs_to :thing, touch: true` on the child model, and then use `after_commit :do_something, on: [:create, :update]` on the parent model.

Let’s update our code to log a message whenever our address is updated:

```ruby
class User < ApplicationRecord
	has_one :address

	after_commit :broadcast_user_updated_event

	private

	def broadcast_user_updated_event
		broadcast(Events::UserUpdated.new(user: self)
	end
end
```

```ruby
class Address < ApplicationRecord
	belongs_to :user, touch: true
end
```

Now our `:user_updated` event will also fire when our user’s address is updated!

## Summary

Wisper is a great library for de-spaghettifying events in your rails apps.

It provides an easy to understand pattern for decoupling code and with a few additions like using classes for events, it can become a powerful event bus for your Rails apps.
