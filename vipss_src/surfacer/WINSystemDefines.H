#ifndef _PLATFORM_DEPENDENT_DEFS_H
#define _PLATFORM_DEPENDENT_DEFS_H

// #if defined(__linux__)
#  include <string.h>
#  include <strings.h>
// #endif

// #if defined(__linux__) || defined(powerpc) || defined(__APPLE__)
#  include <float.h>
#  include <stdio.h>
#  include <stdlib.h>
// #  include <unistd.h>
// #  include <netinet/in.h>
#  include  <cfloat>
#  include <cmath>
#  include <climits>
#  include <fstream>
#  include <iostream>
#  include <string>
// #  include <sys/stat.h>

using namespace std;

#  define UINT    int

#  define EmptyTemplate template<>
#  define tinline 
#  define tempbrackets <>

// This really should be the otherway 'round in windows, but 
// that doesn't seem to work 
#ifndef _isnan
        #define _isnan isnan
#endif
#ifndef _isinf
        #define _isinf isinf
#endif
#ifndef _finite
#define _finite isfinite
#endif

#  ifndef ASSERT
#    include <assert.h>
#    ifndef NDEBUG
#      define ASSERT(f) assert(f)
#      ifndef DEBUG
#        define DEBUG
#      endif
#    else
#      define ASSERT(f)
#    endif
#  endif



/** \brief This gets around the difference in scoping between VC6.0 and .net. 
 Use for all loops after the first one. Usage:

 for ( int i; ... blah ) <br>
 ... <br>
 for ( FORINT i ... blah )

 Second (and subsequent) FORINTs default to either int (new compilers) or
 nothing (old compilers) */
#  define FORINT int
#  define TRACE printf

  extern const char* CTime();

  typedef void (*HHSIG_PF)(int);
  typedef bool WINbool;

#  ifndef FALSE
#    define FALSE 0
#  endif
#  ifndef TRUE
#    define TRUE 1
#  endif

// #endif

/// A kinda clever way to get around undeclared variable warnings in release mode
  /// (use for, e.g., intput index variables that are not touched/used in release mode)
#if defined(DEBUG) || defined(_DEBUG)
#define DECLAREVAR(v) v
#else
#define DECLAREVAR(v)
#endif


template<class T> inline T WINmin(T a, T b) { return a<b?a:b; }
///
template<class T> inline T WINmax(T a, T b) { return a>b?a:b; }
///
template<class T> inline T WINminmax(T a, T tMin, T tMax) 
{ 
    return (a > tMax) ? tMax : ( (a < tMin) ? tMin : a ); 
}

/** @} */

#endif




