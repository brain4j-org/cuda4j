# Cuda4J

**This project is in early development stage!** Expect bugs and issues.

Cuda4J provides **pure Java bindings** for NVIDIA’s **CUDA API**, built on top of the **Foreign Function & Memory (FFM) API** 
introduced in Java 22 and stabilized in Java 25. It enables GPU programming on NVIDIA GPUs directly from Java.

## Features

* Implemented entirely in Java, no JNI dependencies
* Uses the stable Foreign Function & Memory API (Java 25). No JNI
* Wrappers for key CUDA objects: `CudaDevice`, `CudaBuffer`, `CudaModule`, `CudaStream`, `CudaFunction`, `CudaContext`,
  `CudaPointer`

## Requirements

* Java 25 or later
* Gradle or Maven build tool

## Documentation

* [NVIDIA's CUDA API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
* [JEP 454: Foreign Function & Memory API](https://openjdk.org/jeps/454)
* [Java 25 Release Notes](https://openjdk.org/projects/jdk/25/)

## Contributing

Contributions, issues, and pull requests are welcome.
Please open an issue to discuss major changes before submitting a PR.

## License

Cuda4J is licensed under the Apache License, Version 2.0.