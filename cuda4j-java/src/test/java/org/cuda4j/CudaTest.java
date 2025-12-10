package org.cuda4j;

import org.cuda4j.buffer.CudaBuffer;
import org.cuda4j.buffer.CudaPointer;
import org.cuda4j.context.CudaContext;
import org.cuda4j.context.CudaFunction;
import org.cuda4j.context.CudaStream;
import org.cuda4j.device.CudaDevice;
import org.cuda4j.device.CudaModule;
import org.jocl.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.jocl.CL.*;

public class CudaTest {
    
    public static final int N = 512 * 512 * 512;
    public static final int E = 16;
    
    public static void main(String[] args) throws Throwable {
        System.out.println("========= Benchmark for: CUDA =========");
        benchmarkCuda();
        System.out.println("======== Benchmark for: OpenCL ========");
        benchmarkOpenCL();
    }
    
    private static void benchmarkCuda() throws Throwable {
        CudaDevice device = CUDA.createSystemDevice(0);
        
        System.out.println("CUDA Device name: " + device.getName());
        System.out.println("CUDA Device count: " + CUDA.getDeviceCount());
        
        CudaContext context = device.createContext().setCurrent();
        
        CudaModule module = CUDA.loadModule("resources/vector_add.ptx");
        CudaStream stream = CUDA.createStream();
        
        CudaFunction function = module.getFunction("vecAdd");
        
        float[] a = new float[N];
        float[] b = new float[N];
        
        for (int i = 0; i < N; i++) {
            a[i] = i;
            b[i] = i + 1;
        }
        
        CudaBuffer bufA = CUDA.allocateFor(a, N * Float.BYTES);
        CudaBuffer bufB = CUDA.allocateFor(b, N * Float.BYTES);
        CudaBuffer bufC = CUDA.allocateBytes(N * Float.BYTES);
        
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        
        CudaPointer kernelArgs = CudaPointer.from(
            CudaPointer.fromBuffer(bufA),
            CudaPointer.fromBuffer(bufB),
            CudaPointer.fromBuffer(bufC),
            CudaPointer.fromInt(N)
        );
        
        long start = System.nanoTime();
        function.launch(gridSize, 1, 1, blockSize, 1, 1, 0, stream, kernelArgs);
        stream.sync();
        long end = System.nanoTime();
        double took = (end - start) / 1e6;
        
        System.out.println("Took " + took + " millis");
        float[] C = new float[E];
        bufC.copyToHost(C);
        
        System.out.println("Device: " + device.getName());
        System.out.println("Computed on GPU: " + Arrays.toString(C));
    }
    
    private static void benchmarkOpenCL() throws IOException {
        CL.setExceptionsEnabled(true);
        
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);
        
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, null);
        
        cl_device_id device = devices[0];
        
        byte[] deviceNameBytes = new byte[1024];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, Pointer.to(deviceNameBytes), null);
        String deviceName = new String(deviceNameBytes).trim();
        System.out.println(STR."OpenCL Device: \{deviceName}");
        
        cl_context context = clCreateContext(
            null, 1, new cl_device_id[]{device}, null, null, null);
        
        cl_command_queue queue =
            clCreateCommandQueue(context, device, 0, null);
        
        String kernelSource = new String(Files.readAllBytes(Path.of("resources/vector_add.cl")));
        
        cl_program program = clCreateProgramWithSource(context, 1,
            new String[]{kernelSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        
        cl_kernel kernel = clCreateKernel(program, "vecAdd", null);
        
        float[] a = new float[N];
        float[] b = new float[N];
        
        for (int i = 0; i < N; i++) {
            a[i] = i;
            b[i] = i + 1;
        }
        
        cl_mem bufA = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * N, Pointer.to(a), null);
        
        cl_mem bufB = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * N, Pointer.to(b), null);
        
        cl_mem bufC = clCreateBuffer(context,
            CL_MEM_WRITE_ONLY,
            Sizeof.cl_float * N, null, null);
        
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(bufA));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bufB));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(bufC));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{N}));
        
        long globalWorkSize[] = new long[]{N};
        
        long start = System.nanoTime();
        clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, null,
            0, null, null);
        clFinish(queue);
        long end = System.nanoTime();
        
        double took = (end - start) / 1e6;
        System.out.println("Took " + took + " millis");
        
        float[] C = new float[E];
        clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
            Sizeof.cl_float * C.length, Pointer.to(C),
            0, null, null);
        
        System.out.println("Computed on GPU: " + Arrays.toString(C));
        
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
}
