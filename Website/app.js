const express = require('express');
const app = express();

// path module

const path = require('path');

// body parser

const bodyParser= require('body-parser');

// url parser

const urlencodedParser = bodyParser.urlencoded({ extended: true });
app.use(bodyParser.json()); 

// file uploader

const fileupload = require("express-fileupload");
app.use(fileupload());

const {spawn} = require('child_process');

app.use('/',express.static(path.join(__dirname,'views')));
app.set("view engine", "ejs");
app.set("views", path.join(__dirname,'views'));
app.use(bodyParser.urlencoded({ extended: false }));




app.get('/',(req,res) =>{
    res.status(200).render(path.join(__dirname,'/views/index.ejs'),{message:null, result:null});
})

app.post('/',(req,res) =>{
    if(req.files.image){
        let file = req.files.image;
        let filepath = path.join(__dirname,'/images/',req.files.image.name)
        file.mv(filepath, function(error){
            if(error){
                return res.status(500).render(path.join(__dirname,'/views/index.ejs'),{message:"Some error occured! Please try again!", result:null});
            }
        })
        const process = spawn('python', [path.join(__dirname,'/Realtime model.py'), path.join(__dirname,'/images/'), req.files.image.name]);
        process.stdout.on('data', function (data) {
            fruitList = {Lemon: "https://www.apnikheti.com/en/pn/agriculture/horticulture/citrus/lemon", Mango:"https://www.apnikheti.com/en/pn/agriculture/horticulture/fruit/mango", Guava:"https://www.apnikheti.com/en/pn/agriculture/horticulture/fruit/guava", Jamun:"https://www.apnikheti.com/en/pn/agriculture/horticulture/fruit/jamun"}
            const result = data.toString().replace(/(\r\n|\n|\r)/gm, "");
            toSend = []
            toSend.push(result)
            toSend.push(fruitList[result])
            res.status(200).render(path.join(__dirname,'/views/index.ejs'),{message:null, result:toSend});
        })
        process.stderr.on('data', function(data){
            console.log("error: ",data.toString())
        });
    }
    else{
        res.status(200).render(path.join(__dirname,'/views/index.ejs'),{message:"Please upload an image!", result:null});
    }
})

module.exports.app=app

const server = app.listen(3001, ()=> console.log(`Listening on port 3001...   http://localhost:3001`)); 
