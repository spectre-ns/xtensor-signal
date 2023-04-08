#ifndef XTENSOR_SIGNAL_LFILTER_HPP
#define XTENSOR_SIGNAL_LFILTER_HPP

#include <optional>
#include <xtensor.hpp>

namespace xt {
    namespace signal {
        namespace detail{
            /**
             * @brief compute direct form 2 transposed topology
            */
            template<typename E1, typename E2, typename E3, typename E4>
            auto df2_transposed(E1&& b, E2&& a, E3&& x, E4 zi)
            {
                using value_type = typename std::decay_t<E3>::value_type;
                xt::xtensor<value_type, 1> out = xt::zeros<value_type>({ x.shape(0) });

                for (int i = 0; i < x.shape(0); i++)
                {
                    value_type tmp = 0;
                    if (i < zi.shape(0))
                    {
                        tmp = zi(i);
                    }
                    for (int j = 0; j < b.shape(0); j++)
                    {
                        if (!(i - j < 0))
                        {
                            tmp += b(j) * x(i - j);
                        }
                    }

                    for (int j = 1; j < a.shape(0); j++)
                    {
                        if (!(i - j < 0))
                        {
                            tmp -= a(j) * out(i - j);
                        }
                    }

                    tmp /= a(0);
                    out(i) = tmp;
                }
                return out;
            }
        }
        template<class T>
        class lfilter
        {
        public:
            lfilter() : _axis(-1)
            {}

            lfilter& set_axis(std::ptrdiff_t axis)
            {
                _axis = axis;
                return *this;
            }
            template<class E1>
            lfilter& set_zi(E1&& zi)
            {
                _zi = zi;
                return *this;
            }
            template<class E1, class E2>
            lfilter& set_coeffs(E1&& b, E2&& a)
            {
                _a = std::make_optional(a);
                _b = std::make_optional(b);
                return *this;
            }
            template<class E1>
            auto operator()(E1&& x)
            {
                if(!_a.has_value() || !_b.has_value())
                {
                    throw std::runtime_error("Filter coefficients must be set prior to applying filter");
                }
                using value_type = typename std::decay_t<E1>::value_type;
                xt::xarray<value_type> out = xt::zeros<value_type>(x.shape());
                auto saxis = xt::normalize_axis(x.dimension(), _axis);
                auto begin = xt::axis_slice_begin(x, saxis);
                auto end = xt::axis_slice_end(x, saxis);
                auto iter_out = xt::axis_slice_begin(out, saxis);
                for (auto iter = begin; iter != end; iter++)
                {
                    (*iter_out++) = detail::df2_transposed(_b.value(), _a.value(), *iter, _zi);
                }
                return out;
            }
        private:
            std::ptrdiff_t _axis;
            std::optional<xt::xtensor<T, 1>> _a;
            std::optional<xt::xtensor<T, 1>> _b;
            xt::xtensor<T, 1> _zi;
        };
    }
}

#endif