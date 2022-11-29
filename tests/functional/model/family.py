from __future__ import annotations
from dataclay import DataClayObject, activemethod


class Person(DataClayObject):

    name: str
    age: int
    spouse: Person

    @activemethod
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.spouse = None

    @activemethod
    def add_year(self):
        self.age += 1


class Dog(DataClayObject):

    name: str
    puppies: list[Dog]

    @activemethod
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.puppies = []

    @activemethod
    def add_year(self):
        self.age += 1

    @activemethod
    def get_age(self):
        return self.age

    @activemethod
    def new_puppy(self, name):
        puppy = Dog(name, 0)
        self.puppies.append(puppy)
        return puppy


class Family(DataClayObject):

    members: list[Person]

    @activemethod
    def __init__(self):
        self.members = list()

    @activemethod
    def add(self, new_member: Person):
        self.members.append(new_member)

    @activemethod
    def __str__(self) -> str:
        result = ["Members:"]

        for p in self.members:
            result.append(" - Name: %s, age: %d" % (p.name, p.age))

        return "\n".join(result)
