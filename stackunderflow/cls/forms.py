from django import forms

class ClsForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={"class": "text"}))
    
    def clean(self):
        cleaned_data = super(ClsForm, self).clean()
        return cleaned_data
    