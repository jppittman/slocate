//! RAII guard for fast-math floating-point mode (FTZ/DAZ).
//!
//! Denormals hit a microcode slow path on x86 (~100-1000x slower than normal
//! floats). Embedding and dot-product loops are exactly the workloads that
//! generate small residuals that can go subnormal. This guard flushes them
//! to zero for the duration of a scope, then restores the original state.
//!
//! Adapted from pixelflow-core (same technique, same asm).
//!
//! # Platform behavior
//! - **x86_64**: sets FTZ (bit 15) + DAZ (bit 6) in MXCSR
//! - **aarch64**: sets FZ (bit 24) in FPCR
//! - **other**: no-op

pub struct FastMathGuard {
    #[cfg(target_arch = "x86_64")]
    old_mxcsr: u32,
    #[cfg(target_arch = "aarch64")]
    old_fpcr: u64,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    _phantom: (),
}

impl FastMathGuard {
    /// Enable FTZ/DAZ for the current thread.
    ///
    /// # Safety
    /// Modifies global (thread-local) CPU floating-point control state.
    /// Only use where denormal precision loss is acceptable — embedding and
    /// similarity search qualify; general numerics may not.
    #[inline]
    #[must_use]
    pub unsafe fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let old_mxcsr = unsafe { set_mxcsr_fast() };
            Self { old_mxcsr }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let old_fpcr = unsafe { set_fpcr_fast() };
            Self { old_fpcr }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { _phantom: () }
        }
    }
}

impl Drop for FastMathGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            restore_mxcsr(self.old_mxcsr);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            restore_fpcr(self.old_fpcr);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn set_mxcsr_fast() -> u32 {
    // 0x8040 = FTZ (bit 15) | DAZ (bit 6)
    let mut mxcsr: u32 = 0;
    unsafe {
        core::arch::asm!(
            "stmxcsr [{tmp}]",
            "mov {old:e}, [{tmp}]",
            "or dword ptr [{tmp}], 0x8040",
            "ldmxcsr [{tmp}]",
            tmp = in(reg) &mut mxcsr,
            old = out(reg) mxcsr,
            options(nostack),
        );
    }
    mxcsr
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn restore_mxcsr(old: u32) {
    unsafe {
        core::arch::asm!(
            "ldmxcsr [{tmp}]",
            tmp = in(reg) &old,
            options(nostack, readonly),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn set_fpcr_fast() -> u64 {
    let old_fpcr: u64;
    unsafe {
        core::arch::asm!(
            "mrs {old}, fpcr",
            "orr {new}, {old}, #(1 << 24)",
            "msr fpcr, {new}",
            old = out(reg) old_fpcr,
            new = out(reg) _,
            options(nomem, nostack),
        );
    }
    old_fpcr
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn restore_fpcr(old: u64) {
    unsafe {
        core::arch::asm!(
            "msr fpcr, {old}",
            old = in(reg) old,
            options(nomem, nostack),
        );
    }
}
