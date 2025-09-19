package org.cuda4j.device;

import org.cuda4j.CudaObject;
import org.cuda4j.context.CudaContext;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public record CudaDevice(MemorySegment handle, int index) implements CudaObject {
    
    public static final MethodHandle CUDA_DEVICE_NAME = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_device_name"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    public static final MethodHandle CUDA_CREATE_SYSTEM_DEVICE = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_create_system_device"),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT)
    );
    
    public static CudaDevice createSystemDevice(int index) throws Throwable {
        MemorySegment ptr = (MemorySegment) CUDA_CREATE_SYSTEM_DEVICE.invokeExact(index);
        return new CudaDevice(ptr, index);
    }
    
    public String getName() throws Throwable {
        MemorySegment cstrPtr = (MemorySegment) CUDA_DEVICE_NAME.invokeExact(handle);
        return cstrPtr.reinterpret(Long.MAX_VALUE).getString(0);
    }
    
    public CudaContext createContext() throws Throwable {
        return CudaContext.create(this);
    }
}
