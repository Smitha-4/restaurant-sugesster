from django.shortcuts import render
from recommender import recommend

def home(request):
    return render(request, 'index.html')
def analysis(request):
    return render(request, 'analysis.html')
def dashboard1(request):
    return render(request, 'dashboard1.html')
def dashboard2(request):
    return render(request, 'dashboard2.html')
def ml(request):
    return render(request, 'ml.html')
def preprocess(request):
    return render(request, 'preprocess.html')
def sentiment(request):
    return render(request, 'sentiment.html')

def suggestor(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')  # Get input from a form
        recommendations = recommend(input_text)
       
        context = {'recommendations': recommendations}
        return render(request, 'suggestor.html', context)
    else:
        return render(request, 'suggestor.html')  # Initial display