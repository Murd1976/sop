from django.urls import include, path, re_path
from ai_sop import views
from .views import BBLoginView, BBLogoutView

app_name = 'ai_sop'

urlpatterns = [
    #path('', views.index, name='index'),
    
    path('ai/chat/', views.chat_page, name = 'my_sop_chat'),
    path('ai/chat/<int:id>', views.chat_page, name = 'my_sop_chat_id'),
    path('ai/chat/del/<int:id>/', views.delete_chat, name= 'my_del_chat'),
    re_path(r"^(?P<path>.+/|)_create/$", views.Create_Article_Page.as_view(), name = 'my_sop_create_article'),
    
    re_path(r"^ai/sop/(?P<article_id>[0-9]+)/$", views.SopView.as_view(), name = 'my_sop_sop'),
    
    #path('chat/gpt/instruction/chat/', views.chat_instruction, name= 'my_chat_gpt_instruction'),
    #path('chat/gpt/instruction/prop/', views.prop_instruction, name= 'my_prop_gpt_instruction'),
    
   
    #path('chat/db/embedding/chat', views.create_chat_embedding_page, name= 'my_create_chat_embedding'),
    #path('chat/db/embedding/prop', views.create_prop_embedding_page, name= 'my_create_prop_embedding'),
    
    path('accounts/register/activate/<str:sign>/', views.user_activate, name= 'my_register_activate'),
    path('accounts/register/done/', views.RegisterDoneView.as_view(), name= 'my_register_done'),
    path('accounts/register/', views.RegisterUserView.as_view(), name= 'my_register'),
    path('accounts/profile/delete/', views.DeleteUserView.as_view(), name= 'my_profile_delete'),
    path('accounts/profile/change/', views.ChangeUserInfoView.as_view(), name= 'my_profile_change'),
    path('accounts/profile/', views.user_profile, name= 'my_profile'),
    path('accounts/login/', BBLoginView.as_view(), name= 'my_login'),
    path('accounts/logout/', BBLogoutView.as_view(), name= 'my_logout'),
    path('accounts/password/change/', views.BBPasswordChangeView.as_view(), name= 'my_password_change'),
    #path('<str:page>/', views.other_page, name= 'other'),
    
    
    #path('m400', views.m400),
    
     
]



