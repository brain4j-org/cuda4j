package org.cuda4j;

import org.cuda4j.buffer.CudaBuffer;
import org.cuda4j.buffer.CudaPointer;
import org.cuda4j.context.CudaContext;
import org.cuda4j.context.CudaFunction;
import org.cuda4j.context.CudaStream;
import org.cuda4j.device.CudaDevice;
import org.cuda4j.device.CudaModule;

import java.util.Arrays;

public class CudaTest {
    static void main() throws Throwable {
        Cuda4J.init();
        CudaDevice device = CudaDevice.createSystemDevice(0);
        
        System.out.println("CUDA Device name: " + device.getName());
        System.out.println("CUDA Device count: " + Cuda4J.getDeviceCount());
        
        CudaContext context = device.createContext();
        context.setCurrent();
        
        CudaModule module = CudaModule.load("resources/vector_add.ptx");
        CudaFunction function = module.getFunction("vecAdd");
        CudaStream stream = CudaStream.create();
        
        int N = 1024;
        float[] A = new float[N];
        float[] B = new float[N];
        Arrays.fill(A, 1.0f);
        Arrays.fill(B, 2.0f);
        
        CudaBuffer deviceA = CudaBuffer.allocate(A, stream);
        CudaBuffer deviceB = CudaBuffer.allocate(B, stream);
        CudaBuffer deviceC = CudaBuffer.allocate(N * Float.BYTES);
        
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        CudaPointer args = CudaPointer.from(
            CudaPointer.fromBuffer(deviceA),
            CudaPointer.fromBuffer(deviceB),
            CudaPointer.fromBuffer(deviceC),
            CudaPointer.fromInt(N)
        );
        
        function.launch(gridSize, 1, 1, blockSize, 1, 1, 0, stream, args);
        
        float[] hostC = new float[N];
        deviceC.copyToHost(hostC);
        
        System.out.println("Vector add result: " + Arrays.toString(Arrays.copyOf(hostC, 10)));
    }
}
