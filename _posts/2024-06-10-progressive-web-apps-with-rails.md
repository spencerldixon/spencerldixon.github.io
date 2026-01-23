---
layout: post
title: Progressive Web Apps with Rails
description: "Learn how to convert your Rails application into a Progressive Web App (PWA) in 10-15 minutes. Covers manifest.json, service workers, and upcoming Rails 8 PWA defaults."
date: 2024-06-10 10:54 +0100
blog: true
tags: [rails, pwa, mobile]
---
Rails 8 will ship with the files necessary to make your application a Progressive Web App (PWA) by default.

As a Rails developer, I'm a big fan of PWAs. The idea of offering a "good enough" mobile app experience from the same codebase is a huge advantage for small teams and solo developers who don't have the luxury of the time or money needed to maintain a dedicated mobile app codebase.

The good news is that converting your app to a PWA isn't hard. You can do it in about 10-15 minutes. Adding native notifications and offline mode might take you a bit longer, but getting your app to be downloadable to a home screen and function like a mobile app is fairly straightforward.

We can take a look at the [PR](https://github.com/rails/rails/pull/50528/files) for PWA default files in Rails 8 and take inspiration from this to make our Rails <8 apps PWA ready.

## 1. Add the metatags to your layout

```html
<head>
  <meta name="apple-mobile-web-app-capable" content="yes">
  <link rel="manifest" href="/manifest.json">
</head>
```

The MVP for a PWA is to serve a `manifest.json` at the root path of your project. This file contains the config for your PWA, details like the name that should be used for the app, the app icon, a description etc.

## 2. Add a PWA controller to serve the manifest.json

```ruby
class PwaController < ApplicationController
  protect_from_forgery except: :service_worker

  def service_worker
  end

  def manifest
  end
end
```

`service_worker.js` can be thought of as our bridge between our PWA mobile app and our web app. It's capable of doing things like intercepting requests and it's where we'd do things like offline mode should we wish to. It isn't strictly needed for transforming our Rails app into a very basic PWA, but we'll deliver an empty file for now so it's there to expand on later.

## 3. Hook up the routes for the PWA controller

In our `config/routes.rb`

```ruby
get "/service-worker.js" => "pwa#service_worker"
get "/manifest.json" => "pwa#manifest"
```

## 4. Serve our default PWA files

Create a new directory at `app/views/pwa`

We'll add an empty `service_worker.js` here, and then add another file called `manifest.json.erb`.

Copy in the following. Note that since we used the `.json.erb` extension, we can use the `image_path` helper here to pull a 192x192 and 512x512 icon image into the json file. These two image sizes are the bare minimum you need to serve a PWA so make sure your images conform to these sizes and that you have both in the root of your `assets/images` directory.

I've found [this site](https://realfavicongenerator.net/) helpful for creating these app icons. You can upload a high res icon and get back a zip file of icons in the right sizes and formats.

```json
{
  "short_name": "Soju",
  "name": "Soju",
  "id": "/",
  "icons": [
    {
      "src": "<%= image_path 'android-chrome-192x192.png' %>",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "<%= image_path 'android-chrome-512x512.png' %>",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "start_url": "/",
  "background_color": "#fafafa",
  "display": "standalone",
  "scope": "/",
  "theme_color": "#fafafa"
}
```

That's all there is to making your app a very basic PWA!

Commit and host it and you should be able to "Add to Home Screen" when viewing the website in Safari on iOS. The app will save to your home screen, functioning just the same as a mobile app.

It's a great solution for providing the mobile experience without the overhead of maintaining a dedicated mobile codebase.

