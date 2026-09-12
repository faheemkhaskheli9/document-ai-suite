"""Sign-up + login/logout for per-user document/job history (README.md
Section 4: "per-user job/review history"). Login/logout/password-reset
themselves are Django's built-in `django.contrib.auth` views, wired in
`urls.py` -- only sign-up needs a view of our own.
"""
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render


def signup(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("dashboard:home")
    else:
        form = UserCreationForm()

    return render(request, "accounts/signup.html", {"form": form})
