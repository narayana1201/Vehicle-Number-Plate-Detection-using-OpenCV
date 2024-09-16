import React, { useState } from 'react';
import {
    MDBBtn,
    MDBContainer,
    MDBRow,
    MDBCol,
    MDBCard,
    MDBCardBody,
    MDBInput,
    MDBIcon
}
    from 'mdb-react-ui-kit';
import './login.css'
import { useNavigate } from 'react-router-dom';

function Login() {
    const navigate = useNavigate()
    const [formData, setFormData] = useState({
        username: '',
        password: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prevState => ({
            ...prevState,
            [name]: value
        }));
    };
    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('http://127.0.0.1:8000/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            else{
            const responseData = await response.json();
            console.log(responseData);
            navigate('/sidebar/registerpage')
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };
    return (
        <MDBContainer fluid className='login-page '>

            <MDBRow className='d-flex justify-content-center align-items-center h-100'>
                <MDBCol col='12'>

                    <MDBCard className='bg-dark text-white my-5 mx-auto' style={{ borderRadius: '1rem', maxWidth: '400px' }}>
                        <MDBCardBody className='p-5 d-flex flex-column align-items-center mx-auto w-100'>

                            <h2 className="fw-bold mb-2 text-uppercase">Login</h2>
                            <p className="text-white-50 mb-5">Please enter your Id and Password!</p>
                            <form className='form-align' onSubmit={handleSubmit}>

                                <label className='align-self-start'>username</label>
                                <MDBInput wrapperClass='mb-4 mx-5 w-100'
                                    labelClass='text-white'
                                    name='username'
                                    type='string'
                                    size="lg"
                                    value={formData.username}
                                    onChange={handleChange} />
                                <label className='align-self-start'>Password</label>
                                <MDBInput wrapperClass='mb-4 mx-5 w-100'
                                    labelClass='text-white'
                                    name='password'
                                    type='password'
                                    size="lg"
                                    value={formData.password}
                                    onChange={handleChange} />

                                {/* <p className="small mb-3 pb-lg-2"><a class="text-white-50" href="#!">Forgot password?</a></p> */}
                                <MDBBtn className='mx-2 px-5 white-button' color='white' size='lg' type='submit'>
                                    Login
                                </MDBBtn>
                            </form>
                            {/* <div>
                <p className="mb-0">Don't have an account? <a href="#!" class="text-white-50 fw-bold">Sign Up</a></p>

              </div> */}
                        </MDBCardBody>
                    </MDBCard>

                </MDBCol>
            </MDBRow>

        </MDBContainer>
    );
}

export default Login;