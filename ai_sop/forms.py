from django import forms
from django.contrib.auth import password_validation
from django.core.exceptions import ValidationError

from wiki import forms as wiki_forms
from wiki import models as wiki_models
from django.utils.translation import gettext, gettext_lazy as _, pgettext_lazy
from wiki.editors import getEditor

from .models import *
from .apps import user_registered

version = 1.0

class UserForm(forms.Form):
    name = forms.CharField()
    age = forms.IntegerField()
    check1 = forms.ChoiceField(choices=((1, "English"), (2, "German"), (3, "French")), widget=forms.RadioSelect)

class ChangeUserInfoForm(forms.ModelForm):
    email = forms.EmailField(required= True, label= 'Email address')
    
    class Meta:
        model= AdvUser
        fields= ('username', 'email', 'first_name', 'last_name', 'send_messages')
        
class RegisterUserForm(forms.ModelForm):

    email = forms.EmailField(required= True, label= 'Email address')
    password1 = forms.CharField(label = 'Password', widget = forms.PasswordInput,
                               help_text = password_validation.password_validators_help_text_html(), required = True)
    password2 = forms.CharField(label = 'Password (again)', widget = forms.PasswordInput,
                               help_text = 'Enter the password again', required = True)

    def clean_password1(self):
        password1 = self.cleaned_data['password1']
        if password1:
            password_validation.validate_password(password1)
        return password1
    
    def clean(self):
        super().clean()
        if self.is_valid():
            pass2 = self.cleaned_data['password2']
            pass1 = self.cleaned_data['password1']
        
            if pass1 and pass2 and pass1 != pass2:
                errors = {'password2': ValidationError('Passwords do not match', code = 'password_mismatch')}
                raise ValidationError(errors)
        else: 
            errors = {'password1': ValidationError('Validation error', code = 'form error')}
            raise ValidationError(errors)
            
    def save(self, commit = True):
        user = super().save(commit = False)
        user.set_password(self.cleaned_data['password1'])
        user.is_active = False
        user.is_activated = False
        if commit:
            user.save()
        user_registered.send(RegisterUserForm, instance = user)
        return user
    
    class Meta:
        model= AdvUser
        fields= ('username', 'email', 'password1', 'password2', 'first_name', 'last_name', 'send_messages')
        
class ChatForm(forms.Form):
    f_ask_field = forms.CharField(label = 'Ask field', widget= forms.Textarea(attrs={'class':'text_field', 'rows':'8', 'cols':'100'}), disabled = False, required=True)
    f_chat_field = forms.CharField(label = 'Chat field',widget= forms.Textarea(attrs={'class':'text_field', 'rows':'15', 'cols':'100'}), disabled = False, required=False)
    f_debug_field = forms.CharField(label = 'Debug field',widget= forms.Textarea(attrs={'class':'text_field', 'rows':'8', 'cols':'100'}), disabled = False, required=False)
    #f_subject = forms.ChoiceField(label="Select a subject:", initial=0, required= True, choices=((0, "TPS - Contact Form"), (1, "PSC - Contact Form"), (2, "PSN - Contact Form")))
    f_subject = forms.ChoiceField(label="Select a subject:", initial=0, required= True, choices=((0, "DELIVERY INQUIRIES"), (1, "EDIT"), (2, "PRICE MATCHING"), (3, "PRODUCT INQUIRIES"), (4, "RETURNS"), (5, "Customer Service - Guide"), (6, "FAQ")))
    
    f_chat_name = forms.CharField(label="Chat name", initial = 'New', widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 50}))
    
    f_chat_temp = forms.DecimalField(label="Temperature:", min_value=0, max_value=1, initial=0.5, decimal_places=1, widget=forms.NumberInput( attrs={'size':'3'}), required=False)
    
class AiCreateForm(forms.Form, wiki_forms.SpamProtectionMixin):
    def __init__(self, request, urlpath_parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request
        self.urlpath_parent = urlpath_parent

    file_name = forms.ChoiceField(label="Select a file:", initial=0, required= True, choices=((0, "SOP Dry Herbs Vaporizer - V20211031.docx"), (1, "Template Messages - MagicVaporizers.docx"), (2, "FAQ _ Returns DryHerbs.docx")))
    
    title = forms.CharField(
        label=_("Title"),
    )
    slug = wiki_forms.WikiSlugField(
        label=_("Slug"),
        help_text=_(
            "This will be the address where your article can be found. Use only alphanumeric characters and - or _.<br>Note: If you change the slug later on, links pointing to this article are <b>not</b> updated."
        ),
        max_length=wiki_models.URLPath.SLUG_MAX_LENGTH,
    )
    
    instructions = forms.CharField(
        label=_("Instructions"),
        help_text=_("Write your additional instruction for analyze document."),
        required=False,
    )
   

    def clean_slug(self):
        return wiki_forms._clean_slug(self.cleaned_data["slug"], self.urlpath_parent)

    def clean(self):
        self.check_spam()
        return self.cleaned_data
        
class DbLoadForm(forms.Form):
    MY_CHOICES = (
        ('1', 'Add data to embeddings'),
        ('2', 'New embeddings base'),
    )
    
    f_embedding_mode = forms.ChoiceField( label= "Select embedding creation mode:",
        initial = '1',
        choices=MY_CHOICES,
        widget=forms.RadioSelect,
        required=True
    )
    
    db_name = forms.CharField(required=True)
    
class UploadFileForm(forms.ModelForm):
    upload_file = forms.FileField(label= "Upload file:", widget=forms.FileInput(attrs={'title': 'Please, select a file.'}))
    class Meta:
        model = UploadedFile
        fields = ('upload_file',)
    