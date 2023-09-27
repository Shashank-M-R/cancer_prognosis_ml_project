from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver

# Create your models here.

class breast(models.Model):
    breast_photo = models.ImageField(upload_to='images/breast',max_length=255,blank=True,null=True)

@receiver(pre_delete, sender=breast)
def delete_image(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.breast_photo:
        instance.breast_photo.delete(False)


class skin(models.Model):
    skin_photo = models.ImageField(upload_to='images/skin/input',max_length=255,blank=True,null=True)

@receiver(pre_delete, sender=skin)
def delete_image2(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.skin_photo:
        instance.skin_photo.delete(False)


class lung(models.Model):
    lung_photo = models.ImageField(upload_to='images/lung/input/input',max_length=255,blank=True,null=True)

@receiver(pre_delete, sender=lung)
def delete_image3(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.lung_photo:
        instance.lung_photo.delete(False)

from django.core.validators import MaxValueValidator, MinValueValidator

class RiskModel(models.Model):
    air = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    alchol = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    dust = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    occupation = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    genetic = models.IntegerField(default=1,validators=[MaxValueValidator(7), MinValueValidator(1)])
    chronic = models.IntegerField(default=1,validators=[MaxValueValidator(7), MinValueValidator(1)])
    diet = models.IntegerField(default=1,validators=[MaxValueValidator(7), MinValueValidator(1)])
    obesity = models.IntegerField(default=1,validators=[MaxValueValidator(7), MinValueValidator(1)])
    smoking = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    passive_smoke = models.IntegerField(default=1,validators=[MaxValueValidator(8), MinValueValidator(1)])
    chestPain = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    cough_blood = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    fatigue = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])

    age = models.IntegerField(default=1,validators=[MaxValueValidator(13), MinValueValidator(1)])
    race = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    history = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    menarch = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    birth = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    biRads = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    harmone = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    menopause = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    bmi = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(1)])
    biopsy = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    prior = models.IntegerField(default=1,validators=[MaxValueValidator(9), MinValueValidator(0)])
    skin_lesions = models.IntegerField(default=1,validators=[MaxValueValidator(1), MinValueValidator(0)])

