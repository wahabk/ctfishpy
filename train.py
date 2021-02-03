import ctfishpy

unet = ctfishpy.model.Unet('Otoliths')
unet.train()
unet.makeLossCurve()


