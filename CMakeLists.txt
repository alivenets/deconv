cmake_minimum_required(VERSION 3.10)

project(deconv
        VERSION 0.1.0
        LANGUAGES C)

add_library(deconv SHARED deconv.c deconv.h)
set_property(TARGET deconv PROPERTY C_STANDARD 99)