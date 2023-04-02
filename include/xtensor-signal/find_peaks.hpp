#ifndef XTENSOR_SIGNAL_FIND_PEAKS_HPP
#define XTENSOR_SIGNAL_FIND_PEAKS_HPP

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor.hpp>
#include <iostream>
#include <algorithm>
#include <type_traits>

namespace xt {
    namespace signal {
        namespace detail {
            template<
                class E1,
                class E2,
                class E3>
            auto select_by_peak_distance(
                E1&& peaks,
                E2&& priority,
                E3&& distance)
            {
                size_t j, k;
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
                for(auto i : index) 
                {
                    // "Translate" `i` to `j` which points to current peak whose
                    // neighbours are to be evaluated
                    j = priority_to_position[i];
                    if(keep[j] == 0)
                    {
                        // Skip evaluation for peak already marked as "don't keep"
                        continue;
                    }

                    k = j - 1;
                    // Flag "earlier" peaks for removal until minimal distance is exceeded
                    while(0 <= k && peaks[j] - peaks[k] < distance_)
                    {
                        keep[k] = 0;
                        k -= 1;
                    }

                    k = j + 1;
                    // Flag "later" peaks for removal until minimal distance is exceeded
                    while(k < peaks_size && peaks[k] - peaks[j] < distance_)
                    {
                        keep[k] = 0;
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
            template<class E1>
            auto arg_wlen_as_expected(E1&& value)
            {
                //if the value is a none type
                if constexpr (std::is_same<typename std::decay<E1>::type, decltype(xt::xnone())>::value)
                {
                    return -1;
                }
                else
                {
                    //otherwise we have a number
                    //could probably add a check for arithmatic type here
                    if (1 < value)
                    {
                        return std::ceil(value);
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
             * @returns
             * widths The widths for each peak in samples.
             * width_heights The height of the contour lines at which the `widths` where evaluated.
             * left_ips, right_ips 
                    Interpolated positions of left and right intersection points of a
             *     horizontal line at the respective evaluation height.
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
            typename E3 = double,
            typename E4 = decltype(xt::xnone())>
            auto peak_widths(
                E1&& x,
                E2&& peaks,
                E3&& rel_height = .5,
                E4&& wlen = xt::xnone())
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
         * @detail portions of this function are unimplemented and will throw a compiler error
         *     if features requested are not available.
         * @param 1D data vector.
         * @param minimum height thresholds to be filtered. Only minimum heights are implemented.
         * @param threshold unimplemented.
         * @param distance unimplemented.
         * @param prominence unimplemented.
         * @param width defines the minimum width for eligable points. if a second parameter is applied
         *     a max width is also applied.
         * @param wlen defined wlen for calculating prominence.
         * @param rel_height used in defining the width of peaks based on the location. Defaults to .5
         *     or 6dB.
         * @param plateau_size unimplemented.
         */
        template <
            class E1,
            class E2 = decltype(xt::xnone()),
            class E3 = decltype(xt::xnone()),
            class E4 = decltype(xt::xnone()),
            class E5 = decltype(xt::xnone()),
            class E6 = decltype(xt::xnone()),
            class E7 = decltype(xt::xnone()),
            class E8 = double,
            class E9 = decltype(xt::xnone())>
            inline auto find_peaks(
                E1&& x,
                E2&& height = xt::xnone(),
                E3&& threshold = xt::xnone(),
                E4&& distance = xt::xnone(),
                E5&& prominence = xt::xnone(),
                E6&& width = xt::xnone(),
                E7&& wlen = xt::xnone(),
                E8&& rel_height = .5,
                E9&& plateau_size = xt::xnone())
        {
            //check that we have only one dimention
            if (x.shape().size() != 1)
            {
                throw std::runtime_error("Array must be 1D");
            }

            auto [peaks, left_edges, right_edges] = detail::local_maxima_1d(x);

            //check if we want to filter out on height
            if constexpr (std::is_same<typename std::decay<E2>::type, decltype(xt::xnone())>::value == false)
            {
                //to match scipy this
                //we have both parameters
                using data_type = typename std::decay<E1>::type::value_type;
                if (height.shape(0) == 1)
                {
                    //find the values of the peaks
                    xt::xarray<data_type> values = xt::view(x, xt::keep(peaks));

                    //determine which values are below threshold
                    std::transform(values.begin(), values.end(), values.begin(), [&](auto& value) {
                        if (value < height(0))
                        {
                            size_t zero = 0;
                            return static_cast<std::decay<decltype(value)>::type>(zero);
                        }
                        return value;
                        });

                    //find all the indicies that aren't 0
                    std::vector<size_t> peaks_to_keep = xt::nonzero(values).at(0);
                    peaks = xt::view(peaks, xt::keep(peaks_to_keep));
                }
                else if (height.shape(0) == 2)
                {
                    throw std::runtime_error(
                        "Number of height parameters not implemented");
                }
                else if (height.shape(0) == x.shape(0))
                {
                    throw std::runtime_error(
                        "Number of height parameters not implemented");
                }
                else
                {
                    throw std::runtime_error(
                        "Number of height parameters not permitted");
                }

            }
            //check if we want to filter out on threshold
            if constexpr (std::is_same<typename std::decay<E3>::type, decltype(xt::xnone())>::value == false)
            {
                //static_assert(false, "Not implemented");
            }

            //check if we want to filter out on distance
            if constexpr (std::is_same<typename std::decay<E4>::type, decltype(xt::xnone())>::value == false)
            {
                auto keep = detail::select_by_peak_distance(peaks, xt::eval(xt::view(x, xt::keep(peaks))), distance);
                peaks = xt::filter(peaks, keep);
            }
            //check if we want to filter out on prominence
            if constexpr (std::is_same<typename std::decay<E5>::type, decltype(xt::xnone())>::value == false)
            {
                auto wlen_safe = detail::arg_wlen_as_expected(wlen);
                auto [prominences, left_bases, right_bases] = detail::peak_prominences(x, peaks, wlen_safe);
                xt::xarray<bool> keep;
                if (prominence.shape(0) == 1)
                {
                    keep = detail::select_by_property(prominences, prominence(0));
                }
                if (prominence.shape(0) == 2)
                {
                    keep = detail::select_by_property(prominences, prominence(0), prominence(1));
                }
                peaks = xt::filter(peaks, keep);
            }

            //check if we want to filter out on width
            if constexpr (std::is_same<typename std::decay<E6>::type, decltype(xt::xnone())>::value == false)
            {
                //TODO: should probably capture the case of std vector and convert to array
                //once we have the prominence we can add it here to avoid recalculating it again
                auto [widths, width_heights, left_ips, right_ips] = peak_widths(x, peaks, rel_height, wlen);

                //get the indices of interest after applying the filter
                xt::xarray<bool> keep;

                //this can be cleaned up with a wrapper 
                if (width.shape(0) == 1)
                {
                    keep = detail::select_by_property(widths, width(0));
                }
                else if (width.shape(0) == 2)
                {
                    keep = detail::select_by_property(widths, width(0), width(1));
                }
                else
                {
                    throw std::runtime_error(
                        "width length unsupported");
                }
                xt::print_options::set_threshold(100000);
                peaks = xt::filter(peaks, keep);
            }

            //check if we want to filter out on wlen
            if constexpr (std::is_same<typename std::decay<E7>::type, decltype(xt::xnone())>::value == false)
            {
                //static_assert(false, "Not implemented");
            }

            //check if we want to filter out on plateau_size
            if constexpr (std::is_same<typename std::decay<E9>::type, decltype(xt::xnone())>::value == false)
            {
                //static_assert(false, "Not implemented");
            }

            //relative high filter
            return peaks;
        }
    };
};

#endif