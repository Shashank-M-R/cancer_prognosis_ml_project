from django import forms
from .models import breast,lung,skin,RiskModel
from django.contrib.auth.models import User


class breastForm(forms.ModelForm):
    class Meta():
        model = breast
        fields = '__all__'


class lungForm(forms.ModelForm):
    class Meta():
        model = lung
        fields = '__all__'

class skinForm(forms.ModelForm):
    class Meta():
        model = skin
        fields = '__all__'

class riskForm(forms.ModelForm):
    class Meta():
        model = RiskModel
        fields = '__all__'

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta():
        model = User
        fields = ('username','email','password')
        help_texts = {
            'username': None,
        }