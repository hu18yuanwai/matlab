-include .build_debug/Video.d

.build_debug/Video.l: \
	.build_debug/Video.d \

debug/Video: .build_debug/Video.l \
	.build_debug/Video.o \

