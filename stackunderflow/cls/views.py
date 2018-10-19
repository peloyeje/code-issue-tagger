import os

from django.conf import settings
from django.shortcuts import render

from .forms import ClsForm
from .models import Model, Tagger





def classify(request):
 
    if request.method == 'POST':
    
        form = ClsForm(request.POST)
        if form.is_valid():
            sauvegarde = True
            text = form.cleaned_data['text']
            
            path = os.path.join(settings.BASE_DIR, "files/")
            
            inp_model = Model(path+'model.pkl', path+'vocabulary.pkl', path+'tags.pkl')
            tag = Tagger(text, inp_model)
            tag.predict()
            d = tag.decrypt_top_tags()
            l = sorted(d.items(), key=lambda x: x[1], reverse=True)
            tags = [tag[0] for tag in l[:(min([3,len(l)]))]]
            
            return render(request, 'cls/home.html', {
                                                    'form': form, 
                                                    'sauvegarde': sauvegarde,
                                                    'tags': tags,
                                                    })
    else:
        
        form = ClsForm()
        return render(request, 'cls/home.html', locals())