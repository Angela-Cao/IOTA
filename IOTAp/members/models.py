from django.db import models

# Create your models here.

class Customer(models.Model ):
    GENDER_CHOICES = (('Male','Male'),('Female', 'Female') )
    gender = models.CharField(max_length=6, choices=GENDER_CHOICES)
    age = models.IntegerField()
    salary = models.IntegerField()

    def __str__(self):
            return self.gender


class Text(models.Model ):
    # TEXT_CHOICES = (('Input','Input'),('Random', 'Random') )
    # choice = models.CharField(max_length=6, choices=TEXT_CHOICES)
#    age = models.IntegerField()
#    salary = models.IntegerField()
    text = models.TextField(blank=True, null=True)

    def __str__(self):
            return self.choice
     