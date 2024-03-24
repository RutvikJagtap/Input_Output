import base64
from django.shortcuts import render,redirect
from .models import person_collection

# Create your views here.

def login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = person_collection.find_one({"email" : email})
        if user and user["password"]==password :
            user["image"] = base64.b64encode(user["image"]).decode('utf-8')
            user["_id"] = str(user["_id"])
            request.session["user"] = user
            return redirect("user")
    return render(request,"login.html",{})

def register(request):
    if request.method == "POST" :
        firstname = request.POST.get("firstname")
        address = request.POST.get("address")
        image = request.FILES['image'].read()
        password = request.POST.get("password")
        email = request.POST.get("email")
        phone = request.POST.get("phone")

        user_data = {
            "firstname" : firstname,
            "address" : address,
            "image" : image,
            "email" : email,
            "password" : password,
            "phone" : phone
        }
        person_collection.insert_one(user_data)
        return redirect("login")
    
def user(request):
    user = request.session["user"]
    return render(request,"user.html",{"firstname":user["firstname"],"address":user["address"],"image":user["image"],"email":user["email"],"phone":user["phone"],"id":user["_id"]})