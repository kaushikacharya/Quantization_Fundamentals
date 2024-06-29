# Quantization Fundamentals with Hugging Face

## About

This repository contains

- [Course notes](#course-contents)
- [Lab assignments](#assignments)

## Course Info

- [Course URL](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- Instructors:
  - Younes Belkada
  - Marc Sun
- The instructors are Machine Learning Engineers at Hugging Face.

## Course Contents

|#|Lesson    |       Description     |
|-|----------|-----------------------|
|0|[Introduction](./notes/Lesson_0.md)||
|1|[Handling Big Models](./notes/Lesson_1.md)|<ul><li>Memory requirement of large models vis-a-vis GPU hardware</li><li>Model compression techniques apart from quantization</li><li>Idea behind quantization</li></ul>|
|2|[Data Types and Sizes](./notes/Lesson_2.md)|<ul><li>Integer and Floating point data types</li><li>Floating Point Downcasting</li></ul>|
|3|[Loading ML Models with Different Data Types](./notes/Lesson_3.md)|<ul><li>Model downcasting in PyTorch</li><li>Studying impact of loading Generative AI model in half-precision</li></ul>|
|4|[Quantization Theory](./notes/Lesson_4.md)|<ul><li>Linear quantization using `Quanto`</li><li>Compare linear quantization with downcasting</li></ul>|
|5|[Quantization of LLMs](./notes/Lesson_5.md)|<ul><li>Recent SOTA quantization methods</li><li>Benefits of quantized LLMs</li><li>PEFT: LoRA, QLoRA</li></ul>|

## Assignments

|Lesson #|Assignment|Description|
|-|----------|-----------|
|2|[Data Types and Sizes](./notes/Lesson_2.md#notebook)|<ul><li>Common data types</li><li>Floating Point Downcasting</li></ul>|
|3|[Loading ML Models with Different Data Types](./notes/Lesson_3.md#notebook)|<ul><li>Model downcasting from `float32` to `bfloat16`</li><li>Image caption generation using BLIP model and comparing generated text using downcasted model</li></ul>|
|4|[Quantization Theory](./notes/Lesson_4.md#notebook)|<ul><li>Linear quantization using `Quanto`</li></ul>|

## Certificate

- [Course completion certificate](https://learn.deeplearning.ai/accomplishments/48bbc617-1250-41da-a570-ea8735733aff)
- Issued on June 2024

## Related Courses

Please visit my [Github page](https://kaushikacharya.github.io/courses/) for other courses.
