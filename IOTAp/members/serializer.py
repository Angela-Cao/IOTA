from rest_framework import serializers 
from .models import Customer 
from .models import Text 

class CustomerSerializers(serializers.ModelSerializer): 
    class meta: 
        model=Customer 
        fields='__all__'


class TextSerializers(serializers.ModelSerializer): 
    class meta: 
        model=Text 
        fields='__all__'        