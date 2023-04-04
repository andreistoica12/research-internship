# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cpdef process_dict_chunk(dict input_dict):
    cdef dict processed_dict = {}
    for key, index_list in input_dict.items():
        processed_values = process_values(value)  # call your processing function
        if processed_values:
            processed_dict[key] = processed_values
    return dict processed_dict