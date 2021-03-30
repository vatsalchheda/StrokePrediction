import React,{useEffect} from "react"
import TextField from "@material-ui/core/TextField"
import Button from "@material-ui/core/Button"
import "./Login.css"
import img from "./Stroke.png"

function Login(){

    const [username,setUsername] = React.useState(null);
    const [password,setPassword] = React.useState.apply(null)
    return(
        <div className="center">
            <div  className="main">
                <h1>bhavya</h1>
            </div>
        </div>
    )
}

export default Login;