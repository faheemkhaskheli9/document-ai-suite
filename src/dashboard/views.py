from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .registry import all_features


@login_required
def home(request):
    return render(request, "dashboard/home.html", {"features": all_features()})
