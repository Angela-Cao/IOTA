from django import forms
from .models import Customer
from .models import Text


class TextInputForm(forms.Form):
    text_input = forms.CharField(label='Enter text', max_length=100)


class CustomerForm(forms.ModelForm):
    class Meta:
              model = Customer
              fields = "__all__"

    gender = forms.TypedChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    age = forms.IntegerField()
    salary = forms.IntegerField() 


class TextForm(forms.ModelForm):
    class Meta:
              model = Text
              fields = "__all__"

    # choice = forms.TypedChoiceField(choices=[('Input','Input'),('Random', 'Random')])    
    text = forms.TextInput()
    
