from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from .tracker import Tracker 
from django.conf import settings
from django.core.files import File as FileWrapper

class Home(TemplateView) :
    template_name = 'base.html'

def upload(request) :
    context = {}
    if (request.method == 'POST') :

        uploaded_file = request.FILES['document']

        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

        # t = Tracker(name, grit = 10)
        # obj_str = t.getFinish()
        #
        # Peppa Pig
        context['grit'] = 100
        obj_str = '{&quot;0&quot;: {&quot;x&quot;: 399.0, &quot;y&quot;: 426.0, &quot;time&quot;: 200.0}, &quot;1&quot;: {&quot;x&quot;: 371.0, &quot;y&quot;: 482.0, &quot;time&quot;: 200.0}, &quot;2&quot;: {&quot;x&quot;: 397.0, &quot;y&quot;: 428.0, &quot;time&quot;: 200.0}, &quot;3&quot;: {&quot;x&quot;: 11.0, &quot;y&quot;: 482.0, &quot;time&quot;: 300.0}, &quot;4&quot;: {&quot;x&quot;: 363.0, &quot;y&quot;: 176.0, &quot;time&quot;: 400.0}, &quot;5&quot;: {&quot;x&quot;: 384.0, &quot;y&quot;: 288.0, &quot;time&quot;: 400.0}, &quot;6&quot;: {&quot;x&quot;: 419.0, &quot;y&quot;: 371.0, &quot;time&quot;: 1000.0}}'
        context['info'] = obj_str.replace('&quot;', '"')
        
        return render(request, 'view.html', context)

    return render(request, 'upload.html')