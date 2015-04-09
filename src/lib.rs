extern crate num;

use num::traits::*;

use std::ops::*;

#[derive(Copy, Clone, PartialOrd, PartialEq, Debug)]
pub struct Dual<T>(pub T, pub T);

macro_rules! impl_to_primitive {
    ($($name:ident,$ty:ty),*) => {
        impl<T: ToPrimitive> ToPrimitive for Dual<T> {
            $(fn $name(&self) -> Option<$ty> {
                (self.0).$name()
            })*
        }
    }
}

impl_to_primitive!(to_isize, isize, to_i8, i8, to_i16, i16, to_i32, i32, to_i64, i64,
                   to_usize, usize, to_u8, u8, to_u16, u16, to_u32, u32, to_u64, u64,
                   to_f32, f32, to_f64, f64);

impl<N: NumCast> NumCast for Dual<N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let real = NumCast::from(n);
        let dual = NumCast::from(0.0f32);
        if let (Some(real), Some(dual)) = (real, dual) {
            Some(Dual(real,dual))
        } else {
            None
        }
    }
}

impl<T: Neg<Output = T>> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Dual (
            -self.0,
            -self.1,
        )
    }
}

impl<T: Add<Output = T>> Add for Dual<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Dual (
            self.0 + rhs.0,
            self.1 + rhs.1,
        )
    }
}

impl<T: Sub<Output = T>> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Dual (
            self.0 - rhs.0,
            self.1 - rhs.1,
        )
    }
}

impl<T: Mul<Output = T> + Add<Output = T> + Copy> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Dual (
            self.0 * rhs.0,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl<T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy> Div for Dual<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Dual (
            self.0 / rhs.0,
            (self.1 * rhs.0 - self.0 * rhs.1) / (rhs.1 * rhs.1),
        )
    }
}

impl<T: Rem<Output = T> + Copy> Rem for Dual<T> {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Dual (
            self.0 % rhs.0,
            self.1
        )
    }
}

impl<T: Zero> Zero for Dual<T> {
    fn zero() -> Self {
        Dual(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T: Zero + One + Copy> One for Dual<T> {
    fn one() -> Self {
        Dual(T::one(), T::zero())
    }
}

impl<T: Num + Copy> Num for Dual<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;
    fn from_str_radix(string: &str, radix: u32) -> Result<Self, <Self as Num>::FromStrRadixErr> {
        Ok(Dual (
            try!(T::from_str_radix(string, radix)),
            T::zero()
        ))
    }
}

macro_rules! float_impl_basic {
    ($ty:ty, $($name:ident),*) => {
        $(fn $name() -> Self {
            Dual(<$ty as Float>::$name(), <$ty as Zero>::zero())
        })*
    }
}

macro_rules! float_unused_self {
    ($param:ty, $ty:ty, $($name:ident),*) => {
        $(fn $name(_: Option<Self>) -> $ty {
            Float::$name(None::<$param>)
        })*
    }
}

macro_rules! float_impl_passthrough {
    ($result:ty, $($name:ident),*) => {
        $(fn $name(self) -> $result {
            (self.0).$name()
        })*
    }
}

impl<T: Float> Float for Dual<T> {
    float_impl_basic!(T, nan, infinity, neg_infinity, neg_zero,
                      min_value, max_value);
    float_impl_passthrough!(bool, is_nan, is_infinite, is_finite,
                            is_normal, is_sign_positive, is_sign_negative);
    float_impl_passthrough!((u64, i16, i8), integer_decode);
    float_impl_passthrough!(::std::num::FpCategory, classify);

    fn floor(self) -> Self {
        Dual(self.0.floor(), T::zero())
    }

    fn ceil(self) -> Self {
        Dual(self.0.ceil(), T::zero())
    }

    fn round(self) -> Self {
        Dual(self.0.round(), T::zero())
    }

    fn trunc(self) -> Self {
        Dual(self.0.trunc(), T::zero())
    }

    fn fract(self) -> Self {
        Dual(self.0.fract(), self.1)
    }

    fn abs(self) -> Self {
        Dual(self.0.abs(), self.0.signum() * self.1)
    }

    fn signum(self) -> Self {
        Dual(self.0.signum(), T::zero())
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Dual(self.0.mul_add(a.0, b.0), self.1*a.0 + self.0*a.1 + b.1)
    }

    fn recip(self) -> Self {
        Dual::one() / self
    }

    fn powi(self, n: i32) -> Self {
        Dual(self.0.powi(n), T::from(n).unwrap() * self.1.powi(n-1))
    }

    fn powf(self, n: Self) -> Self {
        let real = self.0.powf(n.0);
        Dual(real, n.0*self.0.powf(n.0 - T::one())*self.1 + real*self.0.ln()*n.1)
    }

    fn sqrt(self) -> Self {
        let real = self.0.sqrt();
        Dual(real, self.1 / (T::from(2).unwrap() * real))
    }

    fn exp(self) -> Self {
        let real = self.0.exp();
        Dual(real, self.1 * real)
    }

    fn exp2(self) -> Self {
        let real = self.0.exp2();
        Dual(real, real*self.1*T::from(2).unwrap().ln())
    }

    fn ln(self) -> Self {
        Dual(self.0.ln(), self.1 / self.0)
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        Dual(self.0.log2(), self.1 / (self.0 * T::from(2).unwrap().ln()))
    }

    fn log10(self) -> Self {
        Dual(self.0.log10(), self.1 / (self.0 * T::from(10).unwrap().ln()))
    }

    fn max(self, other: Self) -> Self {
        Dual(self.0.max(other.0), if self.0 >= other.0 { self.1 } else { other.1 })
    }

    fn min(self, other: Self) -> Self {
        Dual(self.0.min(other.0), if self.0 <= other.0 { self.1 } else { other.1 })
    }

    fn abs_sub(self, other: Self) -> Self {
        Dual(
            (self.0 - other.0).max(T::zero()),
            if self.0 > other.0 { self.1 - other.1 } else { T::zero() }
        )
    }

    fn cbrt(self) -> Self {
        let real = self.0.cbrt();
        Dual(real, self.1 / (T::from(3).unwrap() * real*real))
    }

    fn hypot(self, other: Self) -> Self {
        let real = self.0.hypot(other.0);
        Dual(real, (self.0 * other.1 + other.0 * self.1) / real)
    }

    fn sin(self) -> Self {
        Dual(self.0.sin(), self.0.cos()*self.1)
    }

    fn cos(self) -> Self {
        Dual(self.0.cos(), -self.0.sin()*self.1)
    }

    fn tan(self) -> Self {
        let cos = self.0.cos();
        Dual(self.0.tan(), self.1/(cos*cos))
    }

    fn asin(self) -> Self {
        Dual(self.0.asin(), self.1/(T::one() - self.0*self.0).sqrt())
    }

    fn acos(self) -> Self {
        Dual(self.0.acos(), -self.1/(T::one() - self.0*self.0).sqrt())
    }

    fn atan(self) -> Self {
        Dual(self.0.atan(), self.1/(self.0*self.0 + T::one()))
    }

    fn atan2(self, other: Self) -> Self {
        Dual(
            self.0.atan2(other.0),
            (self.0*other.1 - other.0*self.1) / (self.0*self.0 + other.0*other.0)
        )
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self {
        Dual(self.0.exp_m1(), self.1 * self.0.exp())
    }

    fn ln_1p(self) -> Self {
        Dual(self.0.ln_1p(), self.1 / (self.0 + T::one()))
    }

    fn sinh(self) -> Self {
        Dual(self.0.sinh(), self.1 * self.0.cosh())
    }

    fn cosh(self) -> Self {
        Dual(self.0.cosh(), self.1 * self.1.sinh())
    }

    fn tanh(self) -> Self {
        let cosh = self.0.cosh();
        Dual(self.0.tanh(), self.1 / (cosh*cosh))
    }

    fn asinh(self) -> Self {
        Dual(self.0.asinh(), self.1 / (self.0 * self.0 + T::one()).sqrt())
    }

    fn acosh(self) -> Self {
        Dual(self.0.acosh(), self.1 / ((self.0 + T::one()).sqrt() * (self.0 - T::one()).sqrt()))
    }

    fn atanh(self) -> Self {
        Dual(self.0.atanh(), self.1 / (T::one() - self.0 * self.0))
    }
}

pub fn differentiate<T>(fun: fn(Dual<T>) -> Dual<T>) -> Box<Fn(T) -> T>
    where
    T: Float,
{
    Box::new(move |x| fun(Dual(x, T::one())).1)
}

#[test]
fn basic_polynomial() {
    fn poly(x: Dual<i32>) -> Dual<i32> {
        x*x*x + Dual(3,0)*x*x + Dual(10,0)
    }

    assert_eq!(poly(Dual(1,1)), Dual(14,9));
}
