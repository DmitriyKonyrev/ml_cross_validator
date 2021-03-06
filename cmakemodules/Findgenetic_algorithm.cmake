INCLUDE (FindPackageHandleStandardArgs)

SET(GENETIC_ALGORITHM_LIBRARY_DIR "$ENV{MATH_CORE_DIR}")
SET(GENETIC_ALGORITHM_INCLUDE_DIR "$ENV{GENETIC_ALGORITHMS_INCLUDE_DIR}")

FILE(GLOB_RECURSE GENETIC_ALGORITHM_INCLUDE "${GENETIC_ALGORITHM_INCLUDE_DIR}/*.h" "${GENETIC_ALGORITHM_INCLUDE_DIR}/*.hpp")
STRING(REPLACE ";" " " GENETIC_ALGORITHM_INCLUDE ${GENETIC_ALGORITHM_INCLUDE})
STRING(REPLACE "${GENETIC_ALGORITHM_INCLUDE_DIR}/" " " GENETIC_ALGORITHM_INCLUDE ${GENETIC_ALGORITHM_INCLUDE})
STRING(REPLACE "\\" "/" GENETIC_ALGORITHM_INCLUDE ${GENETIC_ALGORITHM_INCLUDE})

 FIND_LIBRARY (GENETIC_ALGORITHM_LIBRARY
  NAMES GENETIC_ALGORITHM
  PATHS "${GENETIC_ALGORITHM_LIBRARY_DIR}"
  PATH_SUFFIXES lib
  DOC "GeneticAlgorithms library")
  
IF (GENETIC_ALGORITHM_INCLUDE)
	IF (GENETIC_ALGORITHM_LIBRARY)
		FIND_PACKAGE_HANDLE_STANDARD_ARGS (GENETIC_ALGORITHM GENETIC_ALGORITHM_INCLUDE GENETIC_ALGORITHM_LIBRARY)
		MESSAGE("WARNING: GENETIC_ALGORITHM_LIBRARY = ${GENETIC_ALGORITHM_LIBRARY}")
		MESSAGE("WARNING: GENETIC_ALGORITHM_INCLUDE = ${GENETIC_ALGORITHM_INCLUDE}")
	ELSE()
		MESSAGE("WARNING: GENETIC_ALGORITHM_LIBRARY = ${GENETIC_ALGORITHM_LIBRARY}")
	ENDIF()
ELSE()
	MESSAGE("WARNING: GENETIC_ALGORITHM_INCLUDE = ${GENETIC_ALGORITHM_INCLUDE}")
endif()