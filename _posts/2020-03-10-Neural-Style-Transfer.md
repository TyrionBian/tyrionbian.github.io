---
layout:            post
title:             "Neural Style Transfer"
date:              2020-03-10
tag:               Computer vision
category:          Computer vision
author:            tianliang
math:              true
---
## Neural Style Transfer

### Overview
In this tutorial, we will learn how to use deep learning to compose images in the style of another image (ever wish you could paint like Picasso or Van Gogh?). This is known as neural style transfer! This is a technique outlined in [Leon A. Gatys' paper, A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), which is a great read, and you should definitely check it out.

But, what is neural style transfer?

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style -- and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

For example, let’s take an image of this turtle and Katsushika Hokusai's The Great Wave off Kanagawa:

<div class="album">
   <figure>
      <img src="{{ "/images/v.jpg" | absolute_url }}" />
      <figcaption>Green_Sea_Turtle_grazing_seagrass</figcaption>
   </figure>   
   <figure>
      <img src="{{ "/images/The_Great_Wave_off_Kanagawa.jpg" | absolute_url }}" />
      <figcaption>The_Great_Wave_off_Kanagawa</figcaption>
   </figure>   
</div>

Now how would it look like if Hokusai decided to paint the picture of this Turtle exclusively with this style? Something like this?

<figure>
<img src="{{ "/images/wave_turtle.png" | absolute_url }}" />
<figcaption>A nice mountain</figcaption>
</figure>




