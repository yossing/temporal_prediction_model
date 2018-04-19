import os
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.cm as cm
import scipy.misc
import numpy as np
import pandas as pd
import pickle as pkl
from bokeh.layouts import row, column, gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Div, ColorBar
from bokeh.models.mappers import LinearColorMapper 
from bokeh.models.widgets import Slider
from bokeh import events
from imp import reload
import visualisation as vis
reload(vis)

SERVER_ADDRESS = '/address/of/server/to/store/weight/images/'
MAIN_SAVE_DIR = '/var/www/html/'


def add_heatmap_colors(in_df, mapping_key):
    norm = matplotlib.colors.Normalize(vmin=in_df[mapping_key].min(), vmax=in_df[mapping_key].max())
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    def map_col(x):
        rgb_col = m.to_rgba(x)
        hex_col = matplotlib.colors.rgb2hex([float(kkk) for kkk in rgb_col[:3]])
        return hex_col

    in_df['color'] = in_df.apply(lambda row: map_col(row[mapping_key]), axis=1)
    return in_df

def get_urls_for_visual_df_data(in_df, x_key, y_key, z_key, RF_size=20, clip_length = 7, 
                                this_save_dir='yossi/test_linked_hover_with_urls/', 
                                root_dir=MAIN_SAVE_DIR, server_address=SERVER_ADDRESS, 
                                verbose=True, order=1):
    new_df = in_df[[x_key, y_key, 'results_path', z_key]].copy()
    new_df = new_df.sort_values(y_key)
    new_df = new_df.sort_values(x_key, ascending=True)
    new_df.reset_index()


    #Now load the results for each input and get the corresponding
    all_urls = []
    # t_step = -1

    # if root_dir is not None:
    save_dir = root_dir + this_save_dir

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    if verbose:
        print('getting weight images...')
    for ii, this_res in  new_df.iterrows():
    #     print(this_res)
        this_res_path = this_res['results_path']
    #     print(this_res_path)
        if 'network_param_values' not in this_res.keys():
            this_net = pkl.load(open(this_res_path, 'rb'))
            network_param_values = this_net.network_param_values
            print('loading network_param_values from file rather than from df')
        else:
            network_param_values = this_res.network_param_values[0]
        weights = network_param_values[0]
        if order == 0:
            weights = weights.T
        weight_image = vis.getVisualWeightImage(weights, RF_size, clip_length, order=order, sort=True, keep_prop=0.01)

        save_name = 'weight_img_'+str(ii) +'.jpeg'
        for t_step in range(clip_length):
            sub_folder = 't_step='+str(t_step+1)+'/'
            if not os.path.exists(os.path.dirname(save_dir+ sub_folder)):
                os.makedirs(os.path.dirname(save_dir+ sub_folder))
            matplotlib.pyplot.imsave(save_dir + sub_folder + save_name, weight_image[:, :, t_step], cmap='gray', origin='lower')

        this_url = server_address + this_save_dir + sub_folder + save_name
        all_urls.append(this_url)

    image_df = pd.DataFrame(dict(url=all_urls), index=new_df.index.values)

    new_df = new_df.join(image_df)
    if verbose:
        print('done')
    new_df = add_heatmap_colors(new_df, z_key)

    new_df['x'] = new_df.apply (lambda row: str(row[x_key]),axis=1)
    new_df['y'] = new_df.apply (lambda row: str(row[y_key]),axis=1)
    new_df['z_value'] = new_df[z_key]
    
    return new_df

def get_urls_for_auditory_df_data(in_df, x_key, y_key, z_key, 
                                  this_save_dir='yossi/test_linked_hover_with_urls/',
                                  root_dir=MAIN_SAVE_DIR, 
                                  server_address=SERVER_ADDRESS, 
                                  verbose=True):

    new_df = in_df[[x_key, y_key, 'results_path', z_key]].copy()

    #Now load the results for each input and get the corresponding 
    all_urls = []


    save_dir = root_dir + this_save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    if verbose:
        print('getting weight images...')
    for ii,this_res in  new_df.iterrows():
        if 'network_param_values' not in this_res.keys():
            print('loading network param values from file')
            this_res_path = this_res['results_path']
        #     print(this_res_path)
            this_net = pkl.load(open(this_res_path, 'rb'))
            if 'network_params' in this_net.__dict__:
                network_param_values = this_net.network_params
            else:
                network_param_values = this_net.network_param_values
        else:
            network_param_values = this_res.network_param_values
        if 't_past' in this_net.input_settings:
            t_past = this_net.input_settings['t_past']
        else:
            t_past = 40#this_net.input_settings['n_h']
        weight_image, weight_image_norm = vis.getWeightImage(network_param_values[0].T,
                                                             this_net.input_settings['numfreq'],
                                                             t_past, 
                                                             keep_prop=0.01)
    
        save_name = 'weight_img_'+str(ii) +'.jpeg'
        h_i_max = np.amax(abs(weight_image_norm.flatten()))
        matplotlib.pyplot.imsave(save_dir + save_name, weight_image_norm, cmap='seismic', vmin=-h_i_max, vmax = h_i_max, origin='lower')
        this_url = server_address + this_save_dir + save_name
        all_urls.append(this_url)

    image_df = pd.DataFrame(dict(url=all_urls), index=in_df.index.values)

    new_df = new_df.join(image_df)
    if verbose:
        print('done')
    new_df = add_heatmap_colors(new_df, z_key)

    new_df['x'] = new_df.apply (lambda row: str(row[x_key]),axis=1)
    new_df['y'] = new_df.apply (lambda row: str(row[y_key]),axis=1)

    new_df['z_value'] = new_df[z_key]
    return new_df


def plot_linked_heatmap_weights(new_df, x_key=None, y_key=None, 
                                save_path=None, add_t_slider=True, 
                                RF_plot_title='Corresponding spatial RFs'):
    # new_df = new_df.sort_index()

    #sort the rows of interest
    if x_key is None and y_key is None:
        x_tick_labels = new_df.sort_values('x', ascending=False)['x'].unique().tolist()
        y_tick_labels = new_df.sort_values('y', ascending=True)['y'].unique().tolist()
    else:
        x_tick_labels = new_df.sort_values(x_key, ascending=True)['x'].unique().tolist()
        y_tick_labels = new_df.sort_values(y_key, ascending=False)['y'].unique().tolist()




    #make the heatmap plot
    s1 = ColumnDataSource(data=new_df)
    p1 = figure(title="Heatmap", tools="hover", toolbar_location='above',
                x_range=x_tick_labels, y_range=y_tick_labels,
                x_axis_label="Log_10 of L1 regularisation on weights",
                y_axis_label='Number of hidden units', 
                width=600, height=600)#, output_backend="webgl")
    r1 = p1.rect('x','y', color='color',source=s1, width=1, height=1)


    #add the colorbar and its label
    colormap = cm.get_cmap("jet") #choose any matplotlib colormap here
    bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    color_mapper = LinearColorMapper(palette=bokehpalette, 
                                     low=np.min(new_df.z_value.values), 
                                     high=np.max(new_df.z_value.values))

    color_bar = ColorBar(color_mapper=color_mapper, location = (1,1))#,
                         # label_standoff=12, border_line_color=None, location=(0,0))
    c_bar_plot = figure(title="Final validation loss", 
                        title_location="right", 
                        height=611, width=140, 
                        toolbar_location=None, 
                        min_border=0, 
                        outline_line_color=None)
    c_bar_plot.add_layout(color_bar, 'right')
    c_bar_plot.title.align = "center"
    

    s2 = ColumnDataSource(data=dict(url=[], x=[],y=[],dw=[],dh=[]))
    s3 = ColumnDataSource(data=new_df)

    p2 = figure(title=RF_plot_title,
                tools=["zoom_in", "zoom_out", "pan", "reset"], 
                x_range=[str(x) for x in np.arange(0, 10)],
                y_range=[str(y) for y in np.arange(0, 10)],
                width=600, height=600, min_border_left=40)

    # turn off x-axis major ticks
    p2.xaxis.major_tick_line_color = None  
    # turn off y-axis major ticks
    p2.yaxis.major_tick_line_color = None  
    r2 = p2.image_url(url='url', x=0, y=0, w='dw', h='dh', source=s2, anchor="bottom_left")

    p1.title.text_font_size = '20pt'
    p1.xaxis.axis_label_text_font_size = "20pt"
    p1.xaxis.major_label_text_font_size = "12pt"
    p1.yaxis.major_label_text_font_size = "12pt"
    p1.yaxis.axis_label_text_font_size = "20pt"
    p2.yaxis.major_label_text_font_size = "0pt"
    p2.xaxis.major_label_text_font_size = "0pt"
    p2.title.text_font_size = '20pt'
    c_bar_plot.title.text_font_size = '16pt'

    if add_t_slider:
        # slider, slider_callback_code = get_t_slider(s2, s3)
        slider = Slider(start=1, end=7, value=7, step=1, 
                        title="Time step", 
                        callback_policy='continuous', 
                        orientation='vertical', 
                        height=200, 
                        bar_color='grey')

        slider_callback_code = """
            //check if the slider value has changed. 
            //If it has, update the timestep of the image being displayed
            var value = cb_obj.value
            if (value === parseInt(value, 10)) {
                //console.log(value)
                //console.log(s2.data)
                var old_url = s2.data.url
                //console.log(old_url)

                if (old_url != []) {
                  old_url=old_url[0]
                  //console.log(old_url)
                  var split_url = old_url.split("=");
                  split_url[1] = split_url[1].replace(split_url[1][0], value);
                  new_url = split_url[0]+"="+split_url[1];
                  var d2 = s2.data;
                  d2['url'] = [new_url]
                  s2.change.emit();
                } 
            }
            """
        def display_event(s2=s2, s3=s3):
            return CustomJS(args=dict(s2=s2, s3=s3), code=slider_callback_code)
        # slider_callback = CustomJS(args=dict(s2=s2, s3=s3), code=slider_callback_code)
        slider.js_on_change('value', display_event())

    else:
        slider = None
        slider_callback_code = ""

    hover = p1.select_one(HoverTool)
    hover.tooltips = None
    hover_callback_code = """
    //console.log(cb_data)
    var indices = cb_data.index['1d'].indices;
    // console.log(indices)
    if (indices.length > 0) {
        //console.log(indices[0])
        //console.log('here!!!')
        //var imgWidth = image.width || image.naturalWidth;
        //var imgHeight = image.height || image.naturalHeight;
        //var img = s3.data.image[indices[0]];
        var url = s3.data.url[indices[0]];
        //console.log(s3.data.image[0][0])
        //console.log(img)

        if (slider != null) {
            var value = slider.value
            var old_url = url
            //console.log(old_url)

            if (old_url != []) {
                if (value === parseInt(value, 10)) {
                    //console.log(value)
                    //console.log(old_url)
                    var split_url = old_url.split("=");
                    split_url[1] = split_url[1].replace(split_url[1][0], value);
                    new_url = split_url[0]+"="+split_url[1];
                    url=new_url
                }
            }
        }

        var d2 = s2.data;
        d2['url'] = [url]
        d2['x'] = [0]
        d2['y'] = [0]
        d2['dw'] = [10]
        d2['dh'] = [10]
        
        s2.change.emit();
        }
    """


    hover.callback = CustomJS(args=dict(s2=s2, s3=s3, slider=slider), 
                              code=hover_callback_code)


    if slider is not None:
        layout = row(p1, c_bar_plot, p2, slider)
    else:
        layout = row(p1, c_bar_plot, p2)

    if save_path is not None:
        output_file(save_path)
    show(layout)


