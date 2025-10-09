---
title: "Granular Polymorphic User Permissions with Cancancan"
layout: post
date: 2022-3-29 16:08
image: '/assets/images/'
description:
tag: rails cancancan permissions authoriziation
blog: true
jemoji:
author:
---

I’ve recently been refactoring how user permissions work in a project to be more granular.

In this project, users are members of organisations, and organisations have many funds and needs.

The manager of the organisation can CRUD these funds and needs, but a regular user should only be able to read them, unless given special permission. This special permission would work on a per-item basis.

Users should also be able to read or manage funds from outside of their organisation if they have been granted special access, by someone within the organisation.

This sounds complex, but the tl;dr is this:

- Members of an organisation can see things inside that organisation
- Managers of the organisation can CRUD things in that organisation
- Granting a special permission between a user and a specific item supersedes any organisation permissions and means that user can access that thing according to whichever read/write role specified in the special permission.

At it’s core, we want to be able to store a record in the database that say “this user has this type of permission to access this item”.

Once we’re storing this in the database, we can use Cancancan to write policies on who should be able to CRUD what, depending on their stored permissions.

## Implementation

First, we’ll create a table to store our user permissions in our database. These records will link a user, with a given accessible item (either a Fund, Need or Organisation), and we’ll also have a column for what type of access they’ll have; read or write.

```ruby
class CreatePermissions < ActiveRecord::Migration[7.0]
  def change
    create_enum :permission_role, ["read", "write"]

    create_table :permissions, id: :uuid do |t|
      t.references :user, null: false, foreign_key: true, type: :uuid
      t.references :accessible, polymorphic: true, null: false, type: :uuid
      t.enum :role, enum_type: :permission_role, default: "read", null: false

      t.timestamps
    end
  end
end
```

We’ll then fill out our model for our new Permissions table

```ruby
class Permission < ApplicationRecord
  belongs_to :user
  belongs_to :accessible, polymorphic: true

  enum role: {
    read: "read",
    write: "write"
  }, suffix: true
end
```

I like to use `suffix: true`, which means rails will generate some helper methods for getting and setting roles made by joining our role and enum names, for example: `permission.read_role?`

Now we can add the other side of our permissions association to our user model, as well as each model we want to make “accessible”.

```ruby
class User < ApplicationRecord
  has_many :organisations, through: :permissions, source: :accessible, source_type: "Organisation"
	has_many :permissions, dependent: :destroy
end
```

```ruby
class Organisation < ApplicationRecord
	has_many :funds
	has_many :needs
	has_many :users, through: :permissions
	has_many :permissions, as: :accessible, dependent: :destroy
end
```

```ruby
class Fund < ApplicationRecord
	belongs_to :organisation
	has_many :permissions, as: :accessible, dependent: :destroy
end
```

```ruby
class Need < ApplicationRecord
	belongs_to :organisation
	has_many :permissions, as: :accessible, dependent: :destroy
end
```

Now we’re set up. You should be able to create a permission record in the console...

```ruby
user = User.first
fund = Fund.first

Permission.create(user: user, accessible: fund)
```

Next we want to define the rules around who can access what. I’m using Cancancan for permissions, which generates an `app/models/ability.rb` file to store our access rules.

```ruby
# frozen_string_literal: true

class Ability
  include CanCan::Ability

  def initialize(user)
    if user.admin?
      can :manage, :all
    else
      can :manage, [Fund, Need] do |accessible|
        # Can manage an accessible item via write permission for the organisation it belongs to
        Permission.find_by(accessible: accessible.organisation, user: user, role: "write")
      end

      can :read, [Fund, Need] do |accessible|
        # Can read an accessible item via write permission for the organisation it belongs to
        Permission.find_by(accessible: accessible.organisation, user: user, role: "read")
      end

      # Can read/manage an item if I have direct permission (overrides org level permissions)
      can :manage, [Fund, Need, Organisation], permissions: { user: user, role: "write" }
      can :read, [Fund, Need, Organisation], permissions: { user: user, role: "read" }
    end
  end
end
```

At the very top level, I let admin level users manage everything (this is just a boolean `admin?` column on the user model).

If a user is not an admin, the first thing I want to check is if they are a member of the organisation for the accessible item they’re trying to do something with.

I have two rules here, one for read level access, and once for write.

Every accessible item that isn’t an organisation can go here as long as they belong to an organisation and we can call `.organisation` on them.

Lastly, we have a read and a write rule for checking direct permissions (a link between a user and an accessible item directly without checking through the organisation), we can add organisations here too since a user can have a direct association with an organisation.

Putting access rules in this order means that if we have an accessible item, we check to see if we are a member of its organisation first.

If not, we check to see if we have a direct special permission with that item, and supersede the organisation level permissions.

I think this is a really elegant solution to complex permissions. It’s a lot of flexibility with surprisingly little code.

You can also add tests in your `spec/models/user_spec.rb` like this. This covers the various combinations of who can access what, and leaves a documentation trail for other developers.

```ruby
require "rails_helper"
require "cancan/matchers"

RSpec.describe User, type: :model do
	describe "abilities" do
    subject(:ability) { Ability.new(user) }

    context "when an admin user" do
      let(:user) { create(:user, admin: true) }

      it { is_expected.to be_able_to(:manage, :all) }
    end

    context "when a manager of an organisation" do
      let(:user) { create(:user, admin: false) }
      let(:organisation) { create(:organisation) }
      let!(:external_organisation) { create(:organisation) }
      let!(:permission) { create(:permission, user: user, accessible: organisation, role: "write") }

      let!(:organisation_fund) { create(:fund, organisation: organisation) }
      let!(:organisation_need) { create(:need, organisation: organisation) }
      let!(:external_fund) { create(:fund, organisation: external_organisation) }
      let!(:external_need) { create(:need, organisation: external_organisation) }

      it { is_expected.to be_able_to(:manage, organisation_fund) }
      it { is_expected.to be_able_to(:manage, organisation_need) }
      it { is_expected.not_to be_able_to(:read, external_fund) }
      it { is_expected.not_to be_able_to(:read, external_need) }

      context "with read permission for an external fund" do
        let!(:permission) { create(:permission, user: user, accessible: external_fund, role: "read") }

        it { is_expected.to be_able_to(:read, external_fund) }
        it { is_expected.not_to be_able_to(:manage, external_fund) }
      end
      context "with write permission for an external fund" do
        let!(:permission) { create(:permission, user: user, accessible: external_fund, role: "write") }

        it { is_expected.to be_able_to(:manage, external_fund) }
      end
    end

    context "without being a member of an organisation" do
      let(:user) { create(:user, admin: false) }
      let(:organisation) { create(:organisation) }
      let!(:fund) { create(:fund, organisation: organisation) }
      let!(:need) { create(:need, organisation: organisation) }

      context "when reading a fund belonging to the organisation" do
        it { is_expected.not_to be_able_to(:read, fund) }
      end

      context "when reading a fund belonging to the organisation" do
        it { is_expected.not_to be_able_to(:read, need) }
      end

      context "with a read permission record" do
        let!(:fund_permission) { create(:permission, user: user, accessible: fund, role: "read") }
        let!(:need_permission) { create(:permission, user: user, accessible: need, role: "read") }

        it { is_expected.to be_able_to(:read, fund) }
        it { is_expected.to be_able_to(:read, need) }
      end

      context "with a write permission record" do
        let!(:fund_permission) { create(:permission, user: user, accessible: fund, role: "write") }
        let!(:need_permission) { create(:permission, user: user, accessible: need, role: "write") }

        it { is_expected.to be_able_to(:manage, fund) }
        it { is_expected.to be_able_to(:manage, need) }
      end
    end
	end
end
```

Lastly, I’m using GraphQL for the API in this application, so to restrict a query, we can use the `.can?` method on our ability class with the current user, fund, and permission we want to check for to return a boolean and a surrounding if statement to decide if we return the query or raise an error.

```ruby
module Queries
  class Fund < Queries::BaseQuery
    description "Find a specific fund"

    argument :id, ID, required: true

    type Types::FundType, null: false

    def ready?(**args)
      authenticate
    end

    def resolve(id:)
      fund = ::Fund.find(id)

      if Ability.new(current_user).can?(:read, fund)
        fund
      else
        unauthorized_error
      end
    rescue ActiveRecord::RecordNotFound => error
      raise GraphQL::ExecutionError.new(error)
    end
  end
end
```
