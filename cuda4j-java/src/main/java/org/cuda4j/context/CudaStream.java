package org.cuda4j.context;

import org.cuda4j.CudaObject;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public record CudaStream(MemorySegment handle) implements CudaObject {
    
    public static final MethodHandle CUDA_STREAM_CREATE = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_stream_create"),
        FunctionDescriptor.of(ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_STREAM_DESTROY = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_stream_destroy"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_STREAM_SYNC = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_stream_sync"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_STREAM_QUERY = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_stream_query"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    
    public static CudaStream create() throws Throwable {
        MemorySegment ptr = (MemorySegment) CUDA_STREAM_CREATE.invoke();
        
        if (ptr == null || ptr.address() == 0) {
            throw new RuntimeException("Failed to create CUDA stream");
        }
        
        return new CudaStream(ptr);
    }
    
    public void sync() throws Throwable {
        int res = (int) CUDA_STREAM_SYNC.invoke(handle);
    
        if (res != 0) {
            throw new RuntimeException("cuStreamSynchronized failed: " + res);
        }
    }
    
    public boolean isCompleted() throws Throwable {
        int res = (int) CUDA_STREAM_QUERY.invoke();
        return res == 0;
    }
}
