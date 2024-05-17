---
layout: post
title: "GSoC - I made it!"
subtitle: "Intro to my GSoC project"
date: 2024-05-17
background: "/img/posts/1.png"
tags: gsoc
---

## What is Open-Source and GSoC?

Open source software is a software with source code that anyone can inspect, modify, and enhance. There are many institutions and individuals who write open software, mainly for research or free deployment purposes. Mostly these softwares, have only a few maintainers, and multiple people, writing and debugging the code, helps a lot. This is where Google Summer of Code `GSoC` comes into the picture. It is a global, online program focused on bringing new contributors into open source software development. Many organisations float projects for the developers to take over the summer and Google mediates in the process, while also paying the contributors for their work over the summer.

## What is my project about?

It has 3 main components:

- Create a Deep Learning Model which accurately identifys commercial segments within video streams.
  - Create a dataset which consists of Commercials and Non-Commercial Videos
  - Design and Train a model architecture which detects commercials in real-time
  - Evaluate the model performance
- Implement a GStreamer plugin
  - Create a custom GStreamer plugin for BeagleBoard that utilizes the trained model to detect commercials in real-time
  - The Plugin replace them with alternative content or obfuscate them, alongside replacing the audio with predefined streams.
- Optimize for BeagleBoard
  - Ensure the entire system is optimized for real-time performance on BeagleBoard hardware.

##
