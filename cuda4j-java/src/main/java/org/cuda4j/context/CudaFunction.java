package org.cuda4j.context;

import org.cuda4j.CudaObject;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public record CudaFunction(MemorySegment handle) implements CudaObject {
    
    private static final MethodHandle CUDA_LAUNCH_KERNEL = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_launch_kernel"),
        FunctionDescriptor.of(
            ValueLayout.JAVA_INT, // return
            ValueLayout.ADDRESS, // function
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // grid
            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, // block
            ValueLayout.JAVA_INT, // shared mem
            ValueLayout.ADDRESS, // stream
            ValueLayout.ADDRESS // kernel params
        )
    );
    
    public int launch(
        int gridX, int gridY, int gridZ,
        int blockX, int blockY, int blockZ,
        int sharedMemBytes,
        MemorySegment stream,
        MemorySegment kernelParams
    ) throws Throwable {
        return (int) CUDA_LAUNCH_KERNEL.invoke(
            handle,
            gridX, gridY, gridZ,
            blockX, blockY, blockZ,
            sharedMemBytes,
            stream == null ? MemorySegment.NULL : stream,
            kernelParams == null ? MemorySegment.NULL : kernelParams
        );
    }
}
