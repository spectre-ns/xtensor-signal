#ifndef XTENSOR_SIGNAL_FIND_PEAKS_HPP
#define XTENSOR_SIGNAL_FIND_PEAKS_HPP

#include <optional>
#include <algorithm>
#include <type_traits>
#include <variant>

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>



namespace xt {
    namespace signal {
        namespace detail {
            template<typename ... Ts>                                                 
            struct Overload : Ts ... { 
                using Ts::operator() ...;
            };
            template<class... Ts> Overload(Ts...) -> Overload<Ts...>;

            template<
                class E1,
                class E2,
                class E3,
                class E4 = decltype(xt::xnone())>
            auto select_by_peak_threshold(E1&& x, E2&& peaks, E3&& tmin, E4&& tmax = xt::xnone())
            {
                //Stack thresholds on both sides to make min / max operations easier :
                //tmin is compared with the smaller, and tmax with the greater thresold to
                //each peak's side
                using value_type = typename std::decay<E1>::type::value_type;
                auto peak_left = xt::eval(peaks - 1);
                auto peak_right = xt::eval(peaks + 1);
                xt::xarray<value_type> left = xt::view(x, xt::keep(peaks)) - xt::view(x, xt::keep(peak_left));
                xt::xarray<value_type> right = xt::view(x, xt::keep(peaks)) - xt::view(x, xt::keep(peak_right));
                auto stacked_thresholds = xt::vstack(xt::xtuple(left, right));
                xt::xarray<bool> keep = xt::ones<bool>({ peaks.size() });
                if constexpr (std::is_same<typename std::decay<E3>::type, decltype(xt::xnone())>::value == false)
                {
                    auto min_thresholds = xt::amin(stacked_thresholds, { 0 });
                    keep = keep && (tmin <= min_thresholds);
                }
                if constexpr (std::is_same<typename std::decay<E4>::type, decltype(xt::xnone())>::value == false)
                {
                    auto max_thresholds = xt::amax(stacked_thresholds, { 0 });
                    keep = keep && (max_thresholds <= tmax);
                }
                return std::make_tuple(keep, stacked_thresholds(0), stacked_thresholds(1));
            }

            template<
                class E1,
                class E2,
                class E3>
            auto select_by_peak_distance(
                E1&& peaks,
                E2&& priority,
                E3&& distance)
            {
                int64_t j, k;
                size_t peaks_size = peaks.shape(0);
                // Round up because actual peak distance can only be natural number
                auto distance_ = std::ceil(distance);
                auto keep = xt::eval(xt::ones<uint8_t>({ peaks_size }));  // Prepare array of flags

                // Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
                // for `peaks` sorted by position).This allows to iterate `peaks`and `keep`
                // with `j` by order of `priority` while still maintaining the ability to
                // step to neighbouring peaks with(`j` + 1) or (`j` - 1).
                auto priority_to_position = xt::argsort(priority);
                auto index = xt::eval(xt::arange<int>(peaks_size - 1, -1, -1));
                // Highest priority first->iterate in reverse order(decreasing)
                for (auto i : index)
                {
                    // "Translate" `i` to `j` which points to current peak whose
                    // neighbours are to be evaluated
                    j = priority_to_position(i);
                    if (keep(j) == 0)
                    {
                        // Skip evaluation for peak already marked as "don't keep"
                        continue;
                    }

                    k = j - 1;
                    // Flag "earlier" peaks for removal until minimal distance is exceeded
                    while (0 <= k && peaks(j) - peaks(k) < distance_)
                    {
                        keep(k) = 0;
                        k -= 1;
                    }

                    k = j + 1;
                    // Flag "later" peaks for removal until minimal distance is exceeded
                    while (k < peaks_size && peaks(k) - peaks(j) < distance_)
                    {
                        keep(k) = 0;
                        k += 1;
                    }
                }
                return keep;
            }


            /**
             * @brief Evaluate where the generic property of peaks confirms to an interval.
             * @param peak_properties
             *     An array with properties for each peak.
             * @param pmin
             *     Lower interval boundary for `peak_properties`. ``None`` is interpreted as
             *     an open border.
             * @param pmax 
             *     Upper interval boundary for `peak_properties`. ``None`` is interpreted as
             *     an open border.
             * @returns
             *     A boolean mask evaluating to true where `peak_properties` confirms to the
             *     interval.
             * @notes Derived from: https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding.py
             */
            template<
                class E1,
                class E2,
                class E3 = decltype(xt::xnone())>
                auto select_by_property(
                    E1&& peak_properties,
                    E2&& pmin,
                    E3&& pmax = xt::xnone())
            {
                xt::xarray<bool> keep = xt::ones<bool>({ peak_properties.shape(0) });

                //if pmin is available
                if constexpr (std::is_same<typename std::decay<E2>::type, decltype(xt::xnone())>::value == false)
                {
                    keep = keep && (pmin <= peak_properties);
                }

                //if pmax is available
                if constexpr (std::is_same<typename std::decay<E3>::type, decltype(xt::xnone())>::value == false)
                {
                    keep = keep && (peak_properties <= pmax);
                }

                return keep;
            }

            /**
             * @brief Calculate the prominence of each peak in a signal.
             * @details The prominence of a peak measures how much a peak stands out from the
             *     surrounding baseline of the signal and is defined as the vertical distance
             *     between the peak and its lowest contour line.
             * @param x sequence A signal with peaks.
             * @param peaks sequence Indices of peaks in `x`.
             * @param wlen int A window length in samples that optionally limits the evaluated area for
             *     each peak to a subset of `x`. The peak is always placed in the middle of
             *     the window therefore the given length is rounded up to the next odd
             *     integer. This parameter can speed up the calculation See notes.
             * @returns
             * prominences The calculated prominences for each peak in `peaks`.
             * left_bases, right_bases
             *     The peaks' bases as indices in `x` to the left and right of each peak.
             *     The higher base of each pair is a peak's lowest contour line.
             * @note Derived from: https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
             */
            template<
                class E1,
                class E2,
                class E3>
                auto peak_prominences(
                    E1&& x,
                    E2&& peaks,
                    E3&& wlen)
            {
                auto prominences = xt::empty<double>({ peaks.shape(0) });
                auto left_bases = xt::empty<int64_t>({ peaks.shape(0) });
                auto right_bases = xt::empty<int64_t>({ peaks.shape(0) });

                for (size_t peak_nr = 0; peak_nr < peaks.shape(0); peak_nr++)
                {
                    auto peak = peaks(peak_nr);
                    auto i_min = 0;
                    auto i_max = x.shape(0) - 1;
                    if (i_min >= peak || peak >= i_max)
                    {
                        throw std::runtime_error(
                            "peak_prominences: Peak is not a valid index for x");
                    }
                    if (2 <= wlen)
                    {
                        i_min = std::max(static_cast<int64_t>(peak - wlen / 2), static_cast<int64_t>(i_min));
                        i_max = std::min(static_cast<int64_t>(peak + wlen / 2), static_cast<int64_t>(i_max));
                    }

                    //find the left bases in interval [i_min, peak]
                    left_bases(peak_nr) = peak;
                    int64_t i = left_bases(peak_nr);
                    auto left_min = x(peak);
                    while (i_min <= i && x(i) <= x(peak))
                    {
                        if (x(i) < left_min)
                        {
                            left_min = x(i);
                            left_bases(peak_nr) = i;
                        }
                        i--;
                    }

                    right_bases(peak_nr) = peak;
                    i = right_bases(peak_nr);
                    auto right_min = x(peak);
                    while (i <= i_max && x(i) <= x(peak))
                    {
                        if (x(i) <= right_min)
                        {
                            right_min = x(i);
                            right_bases(peak_nr) = i;
                        }
                        i++;
                    }

                    prominences(peak_nr) = x(peak) - std::max(left_min, right_min);
                }

                return std::make_tuple(prominences, left_bases, right_bases);
            }

            /**
             * @brief Ensure argument `wlen` is of type `np.intp` and larger than 1.
             * @returns The original `value` rounded up to an integer or -1 if `value` was None.
             * @note Derived from https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding.py
             */
            int arg_wlen_as_expected(std::optional<size_t> value)
            {
                //if the value is a none type
                if(!value.has_value())
                {
                    return -1;
                }
                else
                {
                    //otherwise we have a number
                    //could probably add a check for arithmatic type here
                    if (1 < value.value())
                    {
                        return std::ceil(value.value());
                    }
                }

                throw std::runtime_error("wlen must be greater thank 1");

                return 0;
            }

            /**
             * @brief Calculate the width of each peak in a signal.
             * @details This function calculates the width of a peak in samples at a relative
             *     distance to the peak's height and prominence.
             * @param x A signal with peaks.
             * @param peaks Indices of peaks in `x`.
             * @param rel_height
             *     Chooses the relative height at which the peak width is measured as a
             *     percentage of its prominence. 1.0 calculates the width of the peak at
             *     its lowest contour line while 0.5 evaluates at half the prominence
             *     height. Must be at least 0. See notes for further explanation.
             * @param prominence_data 
             *     A tuple of three arrays matching the output of `peak_prominences` when
             *     called with the same arguments `x` and `peaks`. This data are calculated
             *     internally if not provided.
             * @param wlen 
             *     A window length in samples passed to `peak_prominences` as an optional
             *     argument for internal calculation of `prominence_data`. This argument
             *     is ignored if `prominence_data` is given.
             * @returns widths The widths for each peak in samples. width_heights The height of the contour lines at which the `widths` where evaluated. left_ips, right_ips 
                    Interpolated positions of left and right intersection points of a horizontal line at the respective evaluation height.
             * @note Derived from https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
             */
            template <
                class E1,
                class E2,
                class E3,
                class E4,
                class E5,
                class E6>
                auto peak_widths(
                    E1&& x,
                    E2&& peaks,
                    E3&& rel_height,
                    E4&& prominences,
                    E5&& left_bases,
                    E6&& right_bases)
            {
                if (rel_height < 0)
                {
                    throw std::runtime_error(
                        "Relative height must be equal to or greater than 0");
                }

                if (!(peaks.shape(0) == prominences.shape(0) &&
                    left_bases.shape(0) == right_bases.shape(0) &&
                    prominences.shape(0) == right_bases.shape(0)))
                {
                    throw std::runtime_error(
                        "arrays in prominence_data must have the same shape as peaks");
                }

                auto widths = xt::empty<double>({ peaks.shape(0) });
                auto width_heights = xt::empty<double>({ peaks.shape(0) });
                auto left_ips = xt::empty<double>({ peaks.shape(0) });
                auto right_ips = xt::empty<double>({ peaks.shape(0) });

                for (size_t p = 0; p < peaks.shape(0); p++)
                {
                    size_t i_min = left_bases(p);
                    size_t i_max = right_bases(p);
                    auto peak = peaks(p);

                    //validate the bounds and order
                    if (i_min < 0 || peak < i_min || i_max < peak || i_max >= x.shape(0))
                    {
                        throw std::runtime_error(
                            "Invalid prominence data is invalid for peak");
                    }

                    auto height = x(peak) - prominences(p) * rel_height;
                    width_heights(p) = height;
                    auto i = peak;
                    //find intersecption point on left side
                    while (i_min < i && height < x(i))
                    {
                        i--;
                    }

                    double left_ip = static_cast<double>(i);
                    if (x(i) < height)
                    {
                        left_ip += (height - x(i)) / (x(i + 1) - x(i));
                    }

                    i = peak;
                    while (i < i_max && height < x(i))
                    {
                        i++;
                    }

                    double right_ip = static_cast<double>(i);
                    if (x(i) < height)
                    {
                        right_ip -= (height - x(i)) / (x(i - 1) - x(i));
                    }

                    widths(p) = right_ip - left_ip;
                    left_ips(p) = left_ip;
                    right_ips(p) = right_ip;
                }
                return std::make_tuple(widths, width_heights, left_ips, right_ips);
            }

            /**
             * @brief Find local maxima in a 1D array.
             * @details This function finds all local maxima in a 1D array and returns the indices
             *     for their edges and midpoints (rounded down for even plateau sizes).
             * @param x The array to search for local maxima.
             * @returns Indices of midpoints of local maxima in `x`.
             * left_edges Indices of edges to the left of local maxima in `x`.
             * right_edges Indices of edges to the right of local maxima in `x`.
             * @note Derived from https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
             */
            template<typename T>
            auto local_maxima_1d(T&& x)
            {
                //Preallocate, there can't be more maxima than half the size of `x`
                std::vector<size_t> _midpoints;
                std::vector<size_t> _left_edges;
                std::vector<size_t> _right_edges;

                size_t m = 0;
                size_t i = 1;  // Pointer to current sample, first one can't be maxima
                size_t i_max = x.shape(0) - 1;  //Last sample can't be maxima

                while (i < i_max)
                {
                    if (x(i - 1) < x(i))
                    {
                        auto i_ahead = i + 1;
                        //Find next sample that is unequal to x[i]
                        while (i_ahead < i_max && x(i_ahead) == x(i))
                        {
                            i_ahead++;
                        }
                        //Maxima is found if next unequal sample is smaller than x[i]
                        if (x(i_ahead) < x(i))
                        {
                            _left_edges.push_back(i);
                            _right_edges.push_back(i_ahead - 1);
                            _midpoints.push_back((_left_edges.back() + _right_edges.back()) / 2);
                            m++;
                            i = i_ahead;
                        }
                    }
                    i++;
                }

                //this is weird because you cannot easily append to a xtensor
                xt::xarray<size_t> midpoints = xt::adapt(_midpoints, { _midpoints.size() });
                xt::xarray<size_t> left_edges = xt::adapt(_left_edges, { _left_edges.size() });
                xt::xarray<size_t> right_edges = xt::adapt(_right_edges, { _right_edges.size() });
                return std::make_tuple(midpoints, left_edges, right_edges);
            }
        }

        /**
         * @brief implements the peak widths interface.
         * @param x 1D data vector which matches the data vector used to generate peaks.
         * @param peaks locations of maximums in x. Generally generated from find peaks.
         * @param rel_height defines the height of the peak that is used to generate the width.
         * @param wlen defines the window length used to calculate prominance.
         * @returns tuple of arrays of widths, width heights, left bound and right bounds.
         */
        template<
            typename E1,
            typename E2,
            typename E3 = double>
            auto peak_widths(
                E1&& x,
                E2&& peaks,
                E3&& rel_height,
                std::optional<size_t> wlen)
        {
            //check that we have only one dimention
            if (x.shape().size() != 1)
            {
                throw std::runtime_error(
                    "Array must be 1D");
            }

            if (peaks.shape().size() != 1)
            {
                throw std::runtime_error(
                    "Array must be 1D");
            }

            //declare an internal variable for prominance data
            xt::xarray<double> _prominence_data;
            //check if prominance data is being supplied

            //Calculate prominence if not supplied and use wlen if supplied.
            //check if wlen is an acceptable parameter
            auto wlen_safe = detail::arg_wlen_as_expected(wlen);
            auto [prominences, left_bases, right_bases] = detail::peak_prominences(x, peaks, wlen_safe);
            return detail::peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases);
        }

        /**
         * @brief finds all peaks and applies filters to the peaks based on the parameters provided.
         * @details portions of this function are unimplemented and will throw a compiler error if features requested are not available.
         * @param 1D data vector.
         * @param minimum height thresholds to be filtered. Only minimum heights are implemented.
         * @param threshold unimplemented.
         * @param distance unimplemented.
         * @param prominence unimplemented.
         * @param width defines the minimum width for eligable points. if a second parameter is applied a max width is also applied.
         * @param wlen defined wlen for calculating prominence.
         * @param rel_height used in defining the width of peaks based on the location. Defaults to .5  or 6dB.
         * @param plateau_size unimplemented.
         */
        class find_peaks
        {
        public:
            using height_t = 
            std::variant<
                float,
                std::pair<float, float>, 
                xt::xtensor<float, 1>,
                std::pair<xt::xtensor<float,1>, xt::xtensor<float,1>>
            >;

            using threshold_t =
            std::variant<
                float,
                std::pair<float, float>, 
                xt::xtensor<float, 1>,
                std::pair<xt::xtensor<float,1>, xt::xtensor<float,1>>
            >;

            using prominence_t =
            std::variant<
                float,
                std::pair<float, float>, 
                xt::xtensor<float, 1>,
                std::pair<xt::xtensor<float,1>, xt::xtensor<float,1>>
            >;

            using width_t =
            std::variant<
                float,
                std::pair<float, float>, 
                xt::xtensor<float, 1>,
                std::pair<xt::xtensor<float,1>, xt::xtensor<float,1>>
            >;

            using plateau_t =
            std::variant<
                size_t,
                std::pair<size_t, size_t>, 
                xt::xtensor<size_t, 1>,
                std::pair<xt::xtensor<size_t,1>, xt::xtensor<size_t,1>>
            >;

            find_peaks() : _rel_height(.5)
            {
    
            }
            find_peaks& set_height(height_t height)
            {
                _height = std::make_optional(height);
                return *this;
            }
            find_peaks& set_threshold(threshold_t threshold)
            {
                _threshold = std::make_optional(threshold);
                return *this;
            }
            find_peaks& set_distance(size_t distance)
            {
                _distance = std::make_optional(distance);
                return *this;
            }
            find_peaks& set_prominence(prominence_t prominence)
            {
                _prominence = std::make_optional(prominence);
                return *this;
            }
            find_peaks& set_width(width_t width)
            {
                _width = std::make_optional(width);
                return *this;
            }
            find_peaks& set_wlen(size_t wlen)
            {
                _wlen = std::make_optional(wlen);
                return *this;
            }
            find_peaks& set_rel_height(float rel_height)
            {
                _rel_height = rel_height;
                return *this;
            }
            find_peaks& set_plateau_size(plateau_t plateau_size)
            {
                _plateau_size = std::make_optional(plateau_size);
                return *this;
            }

            template<class E1>
            auto operator()(E1&& x)
            {
                            //check that we have only one dimention
            if (x.shape().size() != 1)
            {
                throw std::runtime_error("Array must be 1D");
            }

            auto all_peaks = detail::local_maxima_1d(x);
            auto peaks = std::get<0>(all_peaks);

            //check if we want to filter out on height
            if(_height.has_value())
            {
                auto peaks_values = xt::view(x, xt::keep(std::get<0>(all_peaks)));
                auto keep = std::visit(detail::Overload{
                        [&](float arg)
                        {
                            return detail::select_by_property(peaks_values, arg);
                        },
                        [&](std::pair<float, float> arg)
                        {
                            return detail::select_by_property(peaks_values, arg.first, arg.second);
                        },
                        [&](xt::xtensor<float, 1> arg)
                        {
                            return detail::select_by_property(peaks_values, arg);
                        },
                        [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg)
                        {
                            return detail::select_by_property(peaks_values, arg.first, arg.second);
                        }
                    }, _height.value());
                peaks = xt::filter(peaks, keep);
            }
            //check if we want to filter out on threshold
            if(_threshold.has_value())
            {
                auto keep = std::visit(detail::Overload{
                        [&](float arg)
                        {
                            auto [keep, left_thresholds, right_thresholds] = detail::select_by_peak_threshold(
                                    x, std::get<0>(all_peaks), arg);
                            return keep;
                        },
                        [&](std::pair<float, float> arg)
                        {
                            auto [keep, left_thresholds, right_thresholds] = detail::select_by_peak_threshold(
                                        x, std::get<0>(all_peaks), arg.first, arg.second);
                            return keep;
                        },
                        [&](xt::xtensor<float, 1> arg)
                        {
                            auto [keep, left_thresholds, right_thresholds] = detail::select_by_peak_threshold(
                                    x, std::get<0>(all_peaks), arg);
                            return keep;
                        },
                        [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg)
                        {
                            auto [keep, left_thresholds, right_thresholds] = detail::select_by_peak_threshold(
                                        x, std::get<0>(all_peaks), arg.first, arg.second);
                            return keep;
                        }
                    }, _threshold.value());

                peaks = xt::filter(peaks, keep);
            }

            //check if we want to filter out on distance
            if(_distance.has_value())
            {
                auto keep = detail::select_by_peak_distance(peaks, xt::eval(xt::view(x, xt::keep(peaks))), _distance.value());
                peaks = xt::filter(peaks, keep);
            }
            //check if we want to filter out on prominence
            if(_prominence.has_value())
            {
                auto wlen_safe = detail::arg_wlen_as_expected(_wlen);
                auto res = detail::peak_prominences(x, peaks, wlen_safe);
                auto keep = std::visit(detail::Overload{
                    [&](float arg)
                    {
                        return detail::select_by_property(std::get<0>(res), arg);
                    },
                    [&](std::pair<float, float> arg)
                    {
                        return detail::select_by_property(std::get<0>(res) , arg.first, arg.second);
                    },
                    [&](xt::xtensor<float, 1> arg)
                    {
                        return detail::select_by_property(std::get<0>(res) , arg);
                    },
                    [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg)
                    {
                       return detail::select_by_property(std::get<0>(res), arg.first, arg.second);
                    }
                }, _prominence.value());
                peaks = xt::filter(peaks, keep);
            }

            //check if we want to filter out on width
            if(_width.has_value())
            {
                //TODO: should probably capture the case of std vector and convert to array
                //once we have the prominence we can add it here to avoid recalculating it again
                auto widths = peak_widths(x, peaks, _rel_height, _wlen);
                auto keep = std::visit(detail::Overload{
                    [&](float arg)
                    {
                        return detail::select_by_property(std::get<0>(widths), arg);
                    },
                    [&](std::pair<float, float> arg)
                    {
                        return detail::select_by_property(std::get<0>(widths), arg.first, arg.second);
                    },
                    [&](xt::xtensor<float, 1> arg)
                    {
                        return detail::select_by_property(std::get<0>(widths), arg);
                    },
                    [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg)
                    {
                       return detail::select_by_property(std::get<0>(widths), arg.first, arg.second);
                    }
                }, _width.value());
                peaks = xt::filter(peaks, keep);
            }

            //check if we want to filter out on plateau_size
            if(_plateau_size.has_value())
            {
                auto plateau_sizes = std::get<1>(all_peaks) -std::get<2>(all_peaks) + 1;
                auto keep = std::visit(detail::Overload{
                    [&](float arg)
                    {
                        return detail::select_by_property(plateau_sizes, arg);
                    },
                    [&](std::pair<float, float> arg)
                    {
                        return detail::select_by_property(plateau_sizes, arg.first, arg.second);
                    },
                    [&](xt::xtensor<float, 1> arg)
                    {
                        return detail::select_by_property(plateau_sizes, arg);
                    },
                    [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg)
                    {
                       return detail::select_by_property(plateau_sizes, arg.first, arg.second);
                    }
                }, _plateau_size.value());
                peaks = xt::filter(peaks, keep);
            }

            return peaks;
        }

        private:
            std::optional<height_t> _height;
            std::optional<threshold_t> _threshold;
            std::optional<size_t> _distance;
            std::optional<prominence_t> _prominence;
            std::optional<width_t> _width;
            std::optional<size_t> _wlen;
            float _rel_height;
            std::optional<plateau_t> _plateau_size;
        };
    }
}

#endif