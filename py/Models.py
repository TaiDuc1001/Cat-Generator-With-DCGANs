from torch import nn

class Generator(nn.Module):
	def __init__(self, image_dim, z_dim, hidden_dim):
		super().__init__()
		def block(in_features, out_features, kernel_size, stride, padding, last_layer=False):
			layers = []
			layers.append(
				nn.ConvTranspose2d(
					in_channels=in_features,
					out_channels=out_features,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding,
					bias=False
				)
			)
			if not last_layer:
				layers.extend([
					nn.BatchNorm2d(num_features=out_features),
					nn.ReLU(inplace=True)
				])
			else:
				layers.append(
					nn.Tanh()
				)
			return layers

		self.model = nn.Sequential(
			# (100, 1, 1)
			*block(z_dim, hidden_dim * 8, 4, 1, 0), # (512, 4, 4)
			*block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1), # (256, 8, 8)
			*block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1), # (128, 16, 16)
			*block(hidden_dim * 2, hidden_dim * 1, 4, 2, 1), # (64, 32, 32)
			*block(hidden_dim, image_dim, 4, 2, 1, last_layer=True) # (3, 64, 64)
		)

	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):
	def __init__(self, image_dim, hidden_dim):
		super().__init__()

		def block(in_features, out_features, kernel_size, stride, padding, first_layer=False, last_layer=False):
			layers = []
			layers.append(
				nn.Conv2d(
					in_channels=in_features,
					out_channels=out_features,
					kernel_size=kernel_size,
					stride=stride,
					padding=padding,
					bias=False)
			)
			if first_layer:
				layers.append(
					nn.LeakyReLU(negative_slope=0.2, inplace=True)
				)

			elif last_layer:
				layers.extend([
					# nn.LeakyReLU(negative_slope=0.2),
          nn.Sigmoid()
        ])

			else:
				layers.extend([
					# nn.Dropout(p=0.8),
					nn.BatchNorm2d(num_features=out_features),
					nn.LeakyReLU(negative_slope=0.2, inplace=True)
				])

			return layers

		self.model = nn.Sequential(
      # (3, 64, 64)
			*block(image_dim, hidden_dim, 4, 2, 1, first_layer=True), # (64, 32, 32)
			*block(hidden_dim * 1, hidden_dim * 2, 4, 2, 1), # (128, 16, 16)
			*block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1), # (256, 8, 8)
			*block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1), # (512, 4, 4)
			*block(hidden_dim * 8, 1, 4, 1, 0, last_layer=True), # (1)
		)

	def forward(self, x):
		return self.model(x)

def weight_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(tensor=m.weight.data, mean=0., std=0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(tensor=m.weight.data, mean=1., std=0.02)
		nn.init.constant_(m.bias.data, 0)