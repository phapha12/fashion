from django.db import models

# Create your models here.
class Prediction(models.Model):
	real = models.CharField(max_length = 20)
	url = models.CharField(max_length = 3000)

	def __str__(self):
		return self.real
