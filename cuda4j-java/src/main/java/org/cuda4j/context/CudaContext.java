package org.cuda4j.context;

import org.cuda4j.CudaObject;
import org.cuda4j.device.CudaDevice;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public record CudaContext(MemorySegment handle) implements CudaObject {
    
    public static final MethodHandle CUDA_CREATE_CONTEXT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_create_context"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_DESTROY_CONTEXT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_destroy_context"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_SYNC_CONTEXT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_sync_context"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT)
    );
    public static final MethodHandle CUDA_CONTEXT_SET_CURRENT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_context_set_current"),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    
    public static CudaContext create(CudaDevice device) throws Throwable {
        MemorySegment ctx = (MemorySegment) CUDA_CREATE_CONTEXT.invoke(device.handle());
        
        if (ctx == null || ctx.address() == 0) {
            throw new RuntimeException("Failed to create CUDA context");
        }
        
        return new CudaContext(ctx);
    }
    
    public void setCurrent() throws Throwable {
        int res = (int) CUDA_CONTEXT_SET_CURRENT.invoke(handle);
        
        if (res != 0) {
            throw new RuntimeException("cuCtxSetCurrent failed: " + res);
        }
    }
    
    public void synchronize() throws Throwable {
        int res = (int) CUDA_SYNC_CONTEXT.invoke();
        
        if (res != 0) {
            throw new RuntimeException("cuCtxSynchronize failed, error " + res);
        }
    }
}
