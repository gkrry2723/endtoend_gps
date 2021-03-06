H = 98
W = 31

kernel_size = (5, 5)
stride = (2, 2)
padding = (0, 0)
dilation = (1, 1)

Hout = ((H + (2 * padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
Wout = ((W + (2 * padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1

print(Hout, Wout)