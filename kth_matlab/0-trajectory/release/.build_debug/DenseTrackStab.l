-include .build_debug/DenseTrackStab.d
-include .build_debug/Saliency.d

.build_debug/DenseTrackStab.l: \
	.build_debug/DenseTrackStab.d \
	.build_debug/Saliency.d \

debug/DenseTrackStab: .build_debug/DenseTrackStab.l \
	.build_debug/DenseTrackStab.o \
	.build_debug/Saliency.o \

