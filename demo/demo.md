<font color="#dd00dd" face>import</font> &nbsp;<font color="#ffffff">spartan</font><br />
<br /><font color="#808080" face="Garamond">#1. Set the computing engine</font><br />
<font color="#ffffff">spartan.config(spartan.engine.<font color="#ff0000">SINGLEMACHINE</font>)</font><br />
<br /><font color="#808080" face="Garamond">#2. Load graph data</font><br />
<font color="#ffffff">data = spartan.SFrame(<font color="#ffff00">"./inputData/yelp"</font>)</font><br />
<br /><font color="#808080" face="Garamond">#3. Create an eigen decomposition model</font><br />
<font color="#ffffff" >edmodel = spartan.eigen_decompose.create(<font color="#ffa500">data</font>, <font color="#ffff00">"eigen decomposition"</font>)</font><br />
<br /><font color="#808080" face="Garamond">#4. Run the model</font><br />
<font color="#ffffff">edmodel.run(spartan.ed_policy.<font color="#008000">SVDS</font>, <font color="#ffa500">k</font><font color="#dd00dd">=</font>10)</font><br />
<br /><font color="#808080" face="Garamond">#5. Show the result</font><br />
<font color="#ffffff">edmodel.showResults()</font><br />
<br /><font color="#ffffff" >admodel = st.anomaly_detection.create(<font color="#ffa500">data</font>, <font color="#ffff00">"anomaly detection"</font>)<br />
admodel.run(spartan.ad_policy.<font color="#008000">FRAUDAR</font>, <font color="#ffa500">k</font><font color="#dd00dd">=</font>3)<br />
admodel.run(spartan.ad_policy.<font color="#008000">HOLOSCOPE</font>, <font color="#ffa500">k</font><font color="#dd00dd">=</font>3)<br /></font><br />