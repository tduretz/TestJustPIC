using CairoMakie, MAT, Printf
using Base.Threads
using JustPIC, JustPIC._3D
using ParallelStencil
using ParallelStencil.FiniteDifferences3D

const backend = JustPIC.CPUBackend 
USE_GPU  = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel_indices (I...) function InitialFieldsParticles!( phases, px, py, pz, index)
    @inbounds for ip in cellaxes(phases)
        # quick escape
        @cell(index[ip, I...]) == 0 && continue
        x = @cell px[ip, I...]
        y = @cell py[ip, I...]
        if x<y
            @cell phases[ip, I...] = 1.0
        else
            @cell phases[ip, I...] = 2.0
        end
    end
    return nothing
end

function main()

    @printf("Running on %d thread(s)\n", nthreads())
    L  = (x=1., y=1., z=1.0)
    Nc = (x=40, y=40, z=40  )
    Nv = (x=Nc.x+1,   y=Nc.y+1,   z=Nc.z+1  )
    Δ  = (x=L.x/Nc.x, y=L.y/Nc.y, z=L.z/Nc.z  )
    Nt   = 40000
    Nout = 1000
    C    = 0.25

    verts     = (x=LinRange(0, L.x, Nv.x), y=LinRange(0, L.y, Nv.y), z=LinRange(0, L.z, Nv.z))
    cents_ext = (x=LinRange(-Δ.x/2, L.x+Δ.x/2, Nc.x+2), y=LinRange(-Δ.y/2, L.y+Δ.y/2, Nc.y+2), z=LinRange(-Δ.z/2, L.z+Δ.z/2, Nc.z+2))
    cents     = (x=LinRange(+Δ.x/2, L.x-Δ.x/2, Nc.x  ), y=LinRange(+Δ.y/2, L.y-Δ.y/2, Nc.y  ), z=LinRange(+Δ.z/2, L.z-Δ.z/2, Nc.z  ))

    # Import data
    file = matopen("data/CornerFlow3D.mat")

    Vx   = read(file, "Vx") # ACHTUNG THIS CONTAINS GHOST NODES AT NORTH/SOUTH
    Vy   = read(file, "Vy") # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
    Vz   = read(file, "Vz") # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
    Pt   = read(file, "Pt")
    close(file)
 
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 12, 50, 5
    particles = init_particles(
        backend, 
        nxcell, 
        max_xcell,
        min_xcell, 
        values(verts),
        values(Δ),
        values(Nc)
    ) # random position by default

    # Initialise phase field
    particle_args = phases, = init_cell_arrays(particles, Val(1))  # cool

    @parallel InitialFieldsParticles!(phases, particles.coords..., particles.index)

    # Time step
    Δt = C * min(Δ...) / max(maximum(abs.(Vx)), maximum(abs.(Vy)))
    @show Δt

    # Create necessary tuples
    V       = (Vx, Vy, Vz)
    grid_vx = (verts.x, cents_ext.y, cents_ext.z)
    grid_vy = (cents_ext.x, verts.y, cents_ext.z)
    grid_vz = (cents_ext.x, cents_ext.y, verts.z)
    Vxc     = 0.5*(Vx[1:end-1,2:end-1,2:end-1] .+ Vx[2:end-0,2:end-1,2:end-1])
    Vyc     = 0.5*(Vy[2:end-1,1:end-1,2:end-1] .+ Vy[2:end-1,2:end-0,2:end-1])
    Vzc     = 0.5*(Vz[2:end-1,2:end-1,1:end-1] .+ Vz[2:end-1,2:end-1,2:end-0])
    Vmag    = sqrt.(Vxc.^2 .+ Vyc.^2 .+ Vyc.^2)

    for it=1:Nt

        # advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), Δt)
        # advection_LinP!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), Δt)
        advection_MQS!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), Δt)
        move_particles!(particles, values(verts), particle_args)        
        # inject_particles!(particles, particle_args, values(verts)) 

        if mod(it,Nout) == 0 || it==1

            @show Npart = sum(particles.index.data)
            particle_density = [sum(p) for p in particles.index]
            @show size(particle_density)

            # Plots
            p = particles.coords
            ppx, ppy, ppz = p
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            pzv = ppz.data[:]
            clr = phases.data[:]
            idxv = particles.index.data[:]
            f = Figure()
            ax1 = Axis(f[1, 1], title="iz= 5", aspect=1.0)
            ax2 = Axis(f[1, 3], title="iz=15", aspect=1.0)
            ax3 = Axis(f[2, 1], title="iz=25", aspect=1.0)
            ax4 = Axis(f[2, 3], title="iz=40", aspect=1.0)

            # f,ax,h=scatter(Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=2)
            # f,ax,h=heatmap(cents.x, cents.y, Vmag[:,:,1])
            # f,ax,h=arrows(cents.x, cents.y, Vxc[:,:,40]./Vmag[:,:,40], Vyc[:,:,40]./Vmag[:,:,20], arrowsize = 5, lengthscale = 1e-2)
            # f,ax,h=arrows(cents.x, cents.z, Vxc[:,:,20]./Vmag[:,:,2], Vzc[:,:,2]./Vmag[:,:,2], arrowsize = 5, lengthscale = 1e-2)
            hm1 = heatmap!(ax1, cents.x, cents.y, particle_density[:,:, 5], colormap=:inferno, colorrange=(0., 35.))
            hm2 = heatmap!(ax2, cents.x, cents.y, particle_density[:,:,15], colormap=:inferno, colorrange=(0., 35.))
            hm3 = heatmap!(ax3, cents.x, cents.y, particle_density[:,:,25], colormap=:inferno, colorrange=(0., 35.))
            hm4 = heatmap!(ax4, cents.x, cents.y, particle_density[:,:,35], colormap=:inferno, colorrange=(0., 35.))
            Colorbar(f[1, 2], hm1)
            Colorbar(f[1, 4], hm2)
            Colorbar(f[2, 2], hm3)
            Colorbar(f[2, 4], hm4)
            display(f)
        end
    end
    return Vx, Vy
end 

Vx, Vy = main()
