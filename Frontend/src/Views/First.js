import React,{useEffect, useState} from "react"
import TextField from "@material-ui/core/TextField"
import Button from "@material-ui/core/Button"
import "./First.css"

 function First(){
    const [result,setresult] = React.useState(null);
    const[isloading,setload] = React.useState(true);
    const [age,setage] = React.useState(null);
    const [state,setState] = React.useState({
        gender:"Male",
        age:age,
        hypertension:0,
        heartdisease:0,
        ever_married:"Yes",
        work_type:"Private",                       

        Residence_type:"Urban",
        avg_glucose:160,
        bmi:27.5,
        smoking_status:"formerly smoked"
    })

function predict(){
    console.log("encountered")
    fetch("/predict",{
        method:"POST",
        body:JSON.stringify({
            gender:"Male",
            age:age,
            hypertension:0,
            heartdisease:0,
            ever_married:"Yes",
            work_type:"Private",
            Residence_type:"Urban",
            avg_glucose:160,
            bmi:27.5,
            smoking_status:"formerly smoked"
        }),
        headers:{
            "Content-Type":"application/json"
        }
    })
    .then((data) => data.json())
    .then(data => {
        setload(false);
        setresult(Number(data))
    })
    .catch(err => console.log(err))

}

function handlechange(e){
    setage(Number(e.target.value));

}
    return(
        <>

            <div className="first">
                <TextField id="standard-basic" label="Standard" onChange={handlechange} value={age} name={age}/>
                <Button variant="outlined" onClick={predict}>Submit Button</Button>
            </div>
            <div>
            {isloading? null   
            :
            (
               <h3>{result}</h3> 
            )

            }
            </div>
        </>
    )
}
export default First;
