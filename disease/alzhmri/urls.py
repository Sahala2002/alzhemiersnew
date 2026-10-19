from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),

    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),

    path("home/", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("stages/", views.stages, name="stages"),

    path("upload/", views.upload_mri, name="upload"),
    path(
        "result/<int:prediction_id>/",
        views.result,
        name="result"
    ),

    path("history/", views.history, name="history"),
    path(
        "recommendations/",
        views.recommendations,
        name="recommendations"
    ),
    path("contact/", views.contact, name="contact"),

    path(
        "admin-dashboard/",
        views.admin_dashboard,
        name="admin_dashboard"
    ),

    path(
        "admin-dashboard/user/<int:user_id>/",
        views.admin_user_activity,
        name="admin_user_activity"
    ),

]