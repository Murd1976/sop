from django.urls import include, path, re_path
from ai_sop.views import wiki_tools, data_tools
from .views.wiki_tools import BBLoginView, BBLogoutView

app_name = 'ai_sop'

urlpatterns = [
    #path('', views.index, name='index'),
    path('ai/chat/data_tools/', data_tools.tools_page, name = 'my_sop_data_tools'),
    path('ai/chat/data_tools/embedding/<str:em_type>/', data_tools.create_embedding_page, name = 'my_sop_embedding'),
    #path('ai/chat/data_tools/chat_embedding/', data_tools.create_chat_embedding_page, name = 'my_sop_chat_embedding'),
    #path('ai/chat/data_tools/prop_embedding/', data_tools.create_prop_embedding_page, name = 'my_sop_prop_embedding'),
    #path('ai/chat/data_tools/del_file/<int:file_id>/', data_tools.delete_files_view, name= 'my_del_data'),
    path('ai/chat/data_tools/upload/', data_tools.upload_file, name='my_upload_file'),
    path('ai/chat/data_tools/del_file/', data_tools.delete_files_view, name='my_del_data'),
    
    path('ai/chat/', wiki_tools.chat_page, name = 'my_sop_chat'),
    path('ai/chat/<int:id>', wiki_tools.chat_page, name = 'my_sop_chat_id'),
    path('ai/chat/del/<int:id>/', wiki_tools.delete_chat, name= 'my_del_chat'),
    re_path(r"^(?P<path>.+/|)_create/$", wiki_tools.Create_Article_Page.as_view(), name = 'my_sop_create_article'),
    
    re_path(r"^ai/assistant/(?P<article_id>[0-9]+)/$", wiki_tools.AssistantView.as_view(), name = 'my_sop_ai_assistant'),
    re_path(r"^ai/sop/(?P<article_id>[0-9]+)/$", wiki_tools.SopView.as_view(), name = 'my_sop_sop'),
    
    #path('chat/gpt/instruction/chat/', views.chat_instruction, name= 'my_chat_gpt_instruction'),
    #path('chat/gpt/instruction/prop/', views.prop_instruction, name= 'my_prop_gpt_instruction'),
    
   
    #path('chat/db/embedding/chat', views.create_chat_embedding_page, name= 'my_create_chat_embedding'),
    #path('chat/db/embedding/prop', views.create_prop_embedding_page, name= 'my_create_prop_embedding'),
    
    path('accounts/register/activate/<str:sign>/', wiki_tools.user_activate, name= 'my_register_activate'),
    path('accounts/register/done/', wiki_tools.RegisterDoneView.as_view(), name= 'my_register_done'),
    path('accounts/register/', wiki_tools.RegisterUserView.as_view(), name= 'my_register'),
    path('accounts/profile/delete/', wiki_tools.DeleteUserView.as_view(), name= 'my_profile_delete'),
    path('accounts/profile/change/', wiki_tools.ChangeUserInfoView.as_view(), name= 'my_profile_change'),
    path('accounts/profile/', wiki_tools.user_profile, name= 'my_profile'),
    path('accounts/login/', BBLoginView.as_view(), name= 'my_login'),
    path('accounts/logout/', BBLogoutView.as_view(), name= 'my_logout'),
    path('accounts/password/change/', wiki_tools.BBPasswordChangeView.as_view(), name= 'my_password_change'),
    #path('<str:page>/', views.other_page, name= 'other'),
    
    
    #path('m400', views.m400),
    
     
]



