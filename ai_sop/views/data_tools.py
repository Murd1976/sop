from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.urls import reverse_lazy, reverse
from django.template import TemplateDoesNotExist
from django.template.loader import get_template
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils.translation import gettext as _

from ..forms import *
from ..ai_sop import *
from ..ai_embedding import *


#company_dir = os.getenv('COMPANY_DIR')


class SOPHandler:
    __data_path = 'ai_sop/db/data/'
    __embeddings_path = 'ai_sop/db/sop/embedding/'
    
    def __init__(self, company_dir):
        self.company_dir = company_dir
        self.work_dir = self.__data_path + self.company_dir
        print(f'\n Path: {self.work_dir} \n')
        
    @property
    def data_path(self):
        return self.__data_path
        
    @data_path.setter
    def data_path(self, f_path):
        if f_path:
            self.__data_path = f_path
            
    
    def delete_data(self, delete_list)->str:
            
            #list_dir = .listdir(path=results_path)
            #print('Existed: ', f_path + delete_list)
            for del_item in delete_list:
                del_item = str(del_item)
                  
                try:
                    print('Loocked for: ', self.work_dir + del_item)
                    if os.path.exists(self.work_dir + del_item):
                        print('Existed: ', self.work_dir + del_item)
                        os.remove(self.work_dir + del_item)
                        #self.list_info.append("Deleted file: " + del_item)
                except:
                    return f"Can't delete file: {del_item}"
                
            
            return 'Ok'

    def file_list(self, f_path = ''):
                    
        f_list = os.listdir(self.work_dir + f_path)
        f_list = [ff for ff in f_list if (os.path.isfile(self.work_dir + f_path + ff) and ('~' not in ff))]
            
        return f_list

    def get_inf_file(self, embeddings_path, f_name):
        with open(embeddings_path + f_name, "r") as f:
            text = f.read()
            
        f_list = text.split('\n')
        f_list = [ff.strip('\n ') for ff in f_list if (ff and (':' not in ff))]
        print(f'Used: {f_list}')    
        return f_list
        
    def is_list_item_in_list(self, list_1, list_2):
        res_dict = {}
        for ff in list_1:        
            if (ff in list_2):
                res_dict[ff] = 'Yes'
            else:
                res_dict[ff] = 'No'
    
        return res_dict

    def delete_files(self, request, files):
                    
        buf_str = files.strip('[]').split(',')
        print(f'Files_id: {buf_str}')
        
        delete_list = []
    #    delete_list.append(files)
        for f_name in buf_str:
            buf = f_name.strip()
            delete_list.append(buf.strip('\''))
        print(delete_list)    
        
        try:                
            res = self.delete_data(delete_list)
            res_text =  _("Files deleted successfully!")
        except Exception as e:
            res_text =  (_("Deleting error: %s") % str(e))
       
        
        return res_text
#------------------------------------------------------------------------------------------------
@login_required
def delete_files_view(request):
    #handler = SOPHandler(company_dir=os.getenv('COMPANY_DIR'))
    buf_str = request.GET.get('ids', '').split(',')
    #print(f'Files_id: {buf_str}')    
    
    try:
        for record_id in buf_str:
            # Удаляем каждую запись по ID                
            up_file = UploadedFile.objects.get(pk=record_id)
            if request.user == up_file.upload_file_owner or request.user.is_staff:    
                                
            
                # Проверяем, является ли пользователь владельцем файла или администратором
                print(up_file.upload_file)
                up_file.upload_file.delete()  # удаляет файл с диска
                up_file.delete()  # удаляет запись из базы данных
                    
            else:
                messages.error(request, _("You do not have permission to delete this file ( %s ).") % str(record))
        messages.success(request, _("Files deleted successfully!"))
        
    except Exception as e:
        messages.error(request, _("Deleting error: %s") % str(e))
            
        
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

@login_required
def tools_page(request):   
    print('URL is working!')
    content = 'The homepage of AI Chat.'
    context = {"content": content, "em_type":''}
    return render(request, "sop/sop_data_tools.html", context)

@login_required
def create_embedding_page(request, em_type="chat"):
    template = 'sop/sop_create_embedding.html'
    
    data_handler = SOPHandler(os.getenv('COMPANY_DIR'))
    
    f_name = "embeddings data"
    
    system_doc = 'support_instruction.txt'
    
    if em_type == "chat":
        gptLearning = WorkerОpenAIChat('chroma', '', data_handler.company_dir)
    else:
        gptLearning = WorkerОpenAI_SOP('chroma', '', data_handler.company_dir)
    
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        selected_files = request.POST.get('selected_files', '').split(',')
        
        #print(f'Files_id: {buf_str}')
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]            
            embedding_mode = userform.cleaned_data['f_embedding_mode']
            gptLearning.em_mode = embedding_mode
            try:
                f_list = []
                for record_id in selected_files:
                    f_list.append(UploadedFile.objects.get(pk=record_id))
                
                if gptLearning.em_mode:
                    unused_list = []
                    for f_ in f_list:
                        buf_str = str(f_.upload_file)
                        unused_list.append( buf_str.split('/')[-1] )
                else:    
                    if em_type == "chat":
                        un_rec = UploadedFile.objects.filter(is_used_chat = False)
                    else:
                        un_rec = UploadedFile.objects.filter(is_used_prop = False)
                    un_list = []
                    for f_ in un_rec:
                        un_list.append(f_.upload_file)
                        
                    unused_list = []
                    for f_ in f_list:
                        if f_.upload_file in un_list:
                            buf_str = str(f_.upload_file)
                            unused_list.append( buf_str.split('/')[-1] )
                #Подготовка эмбедингов
                
                used_list = gptLearning.create_embedding(gptLearning.data_directory, # путь к учебным материалам
                                           unused_list ) # list not used files
                                         
                if request.user.is_superuser:
                    log = ' '.join([x.replace('\n', '<br>') for x in gptLearning.debug_log])
                    messages.success(request, mark_safe(_("Embeddings created successfully! %s") % log))
                    #print(f'Log: {gptLearning.debug_log}')
                else:
                    messages.success(request, _("Embeddings created successfully!"))
            except Exception as e:
                messages.error(request, _("Embeddings error: %s") % str(e))
        
        try:    
            if gptLearning.em_mode:
                
                f_list = UploadedFile.objects.all()
                
            else:
                if em_type == "chat":
                    f_list = UploadedFile.objects.filter(is_used_chat = False)
                else:
                    f_list = UploadedFile.objects.filter(is_used_prop = False)
                
            for rec in f_list:
                #print(rec.upload_file)
                buf_str = str(rec.upload_file)
                if (buf_str.split('/')[-1] in used_list):
                    if em_type == "chat":
                        rec.is_used_chat = True
                    else:
                        rec.is_used_prop = True
                else:
                    if em_type == "chat":
                        rec.is_used_chat = False
                    else:
                        rec.is_used_prop = False
                rec.save()
                    #print(rec.upload_file)
        except Exception as e:
            messages.error(request, _("Error: %s") % str(e))
            
    parts = DbLoadForm(initial= {"db_name":f_name})
           
    f_list = UploadedFile.objects.all()        
            
    context = {"form": parts, "user_name":request.user, "em_type":em_type, "data_list": f_list}
    return render(request, template, context)
   
    
@login_required
def load_chat_data_page(request):

    f_name = "train_data_ask.jsonl"
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            load_from_xls(f_name, request.user)
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'chat/chat_load_chat_base.html'
    chain_list = MessageChain.objects.filter(owner=request.user)

    context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    return render(request, template, context)

@login_required
def load_base_page(request):
    f_name = "chat/upwork_list.xlsx"
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            load_from_xls(f_name, request.user)
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'chat/chat_load_base.html'
    chain_list = MessageChain.objects.filter(owner=request.user)
#    tests_list = AllBackTests.objects.all()
#    tests_list.delete()
#    tests_list = AllBackTests.objects.all()
    context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    return render(request, template, context)

def load_web_page(request):
    res_f_name = 'magicvaporizers.txt'
    my_url = "https://magicvaporizers.co.uk"
    
    #res_f_name = 'magicvaporizers.txt'
    #my_url = "https://magicvaporizers.co.uk"
    
    company_dir = 'MagicVaporizers/'
    #company_dir = 'Maxximize/'
    
    gptLearning = WorkerОpenAI_SOP('', company_dir)
    
    #text = scrape_site(my_url, res_f_name)
    #text = gptLearning.web_text_filter_2('magicvaporizers_short.txt')
    text = gptLearning.doc_analyzer_fields()
    #text = gptLearning.doc_analyzer_sop('FAQ _ Returns DryHerbs.docx')
    parts = EditInstructionForm(initial = {'f_chat_instruction': text})
    template = 'chat/chat_load_web.html'
    #context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    context = {"form": parts, "user_name":request.user}
    return render(request, template, context)

@login_required    
def upload_file(request):
    template = 'sop/sop_upload_data.html'
    
    #data_handler = SOPHandler(os.getenv('COMPANY_DIR'))
    
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.upload_file_owner = request.user
            try:
                uploaded_file.save()
                messages.success(request, _("Files uploaded successfully!"))
            except Exception as e:
                messages.error(request, _("Uploading error: %s") % str(e))
            
    else:
        form = UploadFileForm()
    
    f_list = {}
    try:
        if request.user == request.user.is_staff:
            f_list = UploadedFile.objects.all()
        else:
            f_list = UploadedFile.objects.filter(upload_file_owner = request.user)
    except Exception as e:
            messages.error(request, _("Error: %s") % str(e))
            
    context = {'form': form, 'data_list': f_list}
    return render(request, template, context)
    

