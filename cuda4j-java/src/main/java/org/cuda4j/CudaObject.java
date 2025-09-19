package org.cuda4j;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public interface CudaObject {
    
    Linker LINKER = Cuda4J.LINKER;
    SymbolLookup LOOKUP = Cuda4J.LOOKUP;
    MethodHandle CUDA_RELEASE_OBJECT = LINKER.downcallHandle(
        LOOKUP.findOrThrow("cuda_release_object"),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );

    default void release() throws Throwable {
        CUDA_RELEASE_OBJECT.invokeExact(handle());
    }

    MemorySegment handle();
}