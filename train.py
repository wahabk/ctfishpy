import ctfishpy

unet = ctfishpy.model.Unet()
unet.train()
unet.makeLossCurve()