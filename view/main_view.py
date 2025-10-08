from ipywidgets import *
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

class MainView:
    def __init__(self):
            self.dt_features = None
            self.feat_page = None
            self.process_page = None
            self.vis_page = None
            self.right_page = None
            self.process_types = None
            self.f_box = None
            self.trg_lbl = None
            self.dt_ftslay = None
            self.ftlaycl = None
            self.prdtsk_lbl = None
            self.datasets = None
            self.tab_set = None
            self.tabs = None
            self.cleaningoutput = None

            
    def set_tabs(self,tab_1,tab_2,tab_3,tab_4, tab_5):
        self.tabs = [tab_1,tab_2,tab_3, tab_4, tab_5]
        tab_set = widgets.Tab(self.tabs)
        tab_set.set_title(0, 'Data Selection')
        tab_set.set_title(1, 'Data Cleaning')
        tab_set.set_title(2, 'Data Processing')
        tab_set.set_title(3, 'Predictive Modeling')
        tab_set.set_title(4, 'Logging')
        tab_set.layout.width='100%'
        tab_set.layout.display = 'none' 
        self.tab_set = tab_set

    def show_tabs(self):
         self.tab_set.layout = widgets.Layout(visibility = 'visible')

    def get_tabs(self):
         return self.tab_set

    def get_ui(self, login, task_selection, tabs):
        main_vbox = VBox([login, task_selection, tabs],
                         layout=Layout(flex='1', display='flex', width='100%'))

        ui = HBox([main_vbox],
                  layout=Layout(width='100%', display='flex'))
        return ui
    
    def close_tab(self, index):
         self.tabs[index].close()
     
    def set_title(self, index, titlename):
         self.tab_set.set_title(index, titlename)

    def get_result_bar_chart(self,competence_vector, difficulty_data):
          if competence_vector == None:
               return "No data"
          skills = list(difficulty_data.keys())
          scores = [competence_vector.get(skill, 0) for skill in skills]
          difficulties = [difficulty_data[skill] for skill in skills]
          y_pos = np.arange(len(skills))

          fig, ax = plt.subplots(figsize=(3, 2.5))
          ax.barh(y_pos, scores, color='steelblue', label='Your score')
          for i, diff in enumerate(difficulties):
               ax.plot([diff, diff], [i - 0.4, i + 0.4], color='red', linewidth=2)

          ax.set_yticks(y_pos)
          ax.set_yticklabels(skills, fontsize=8)
          ax.set_xlim(0, 100)
          ax.set_xlabel("Score", fontsize=9)
          ax.set_title("Results", fontsize=10)
          ax.invert_yaxis()

          handles = [
               plt.Line2D([], [], color='red', linewidth=2, label='Difficulty (max score)'),
               plt.Rectangle((0, 0), 1, 1, color='steelblue', label='Your score')
          ]
          fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7)
          plt.tight_layout()

          buf = BytesIO()
          plt.savefig(buf, format='png', bbox_inches='tight')
          buf.seek(0)
          import base64
          img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
          img_html = f'<img src="data:image/png;base64,{img_b64}" width="300"/>'
          plt.close(fig)
          return img_html